/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "device/metal/osl_camera.h"
#  include "device/metal/osl_camera_translate.h"

#  include "kernel/types.h"

#  include "kernel/tables.h"

#  include "util/map.h"
#  include "util/md5.h"
#  include "util/string.h"
#  include "util/unique_ptr.h"
#  include "util/xml.h"

CCL_NAMESPACE_BEGIN

static bool osl_camera_param_convert(const TypeDesc &type, OSLCameraTranslateParam &param)
{
  if (type.arraylen != 0) {
    return false;
  }
  if (type.basetype == TypeDesc::STRING) {
    param.type = OSLCameraTranslateParam::STRING;
    param.size = 0;
    return true;
  }
  if (type.basetype == TypeDesc::INT) {
    param.type = OSLCameraTranslateParam::INT;
  }
  else if (type.basetype == TypeDesc::FLOAT) {
    param.type = OSLCameraTranslateParam::FLOAT;
  }
  else {
    return false;
  }
  param.size = int(type.aggregate);
  return true;
}

#  ifdef WITH_PUGIXML
/* The dictionaries of dict_find(): XML documents queried with XPath, numbered and linked like
 * in the OSL runtime. */
class OSLCameraXMLDictionary : public OSLCameraDictionary {
 public:
  OSLCameraXMLDictionary()
  {
    /* Node 0 is no node. */
    nodes_.push_back({});
  }

  int find(const std::string &dictionary, const std::string &query) override
  {
    const int document = document_index(dictionary);
    if (document < 0) {
      return -1;
    }
    return find(std::make_pair(-document - 1, query), xml_node(*documents_[document]));
  }

  int find(const int node, const std::string &query) override
  {
    if (node <= 0 || node >= int(nodes_.size())) {
      return 0;
    }
    return find(std::make_pair(node, query), nodes_[node].node);
  }

  int next(const int node) override
  {
    return (node <= 0 || node >= int(nodes_.size())) ? 0 : nodes_[node].next;
  }

  bool value(const int node, const std::string &attribute, std::string &text) override
  {
    if (node <= 0 || node >= int(nodes_.size())) {
      return false;
    }
    const xml_node &xml = nodes_[node].node;
    if (attribute.empty()) {
      text = xml.value();
      return true;
    }
    for (const xml_attribute &attr : xml.attributes()) {
      if (attribute == attr.name()) {
        text = attr.value();
        return true;
      }
    }
    return false;
  }

  int num_nodes() override
  {
    return int(nodes_.size()) - 1;
  }

 private:
  struct Node {
    xml_node node;
    int next = 0;
  };

  int document_index(const std::string &dictionary)
  {
    const auto it = document_map_.find(dictionary);
    if (it != document_map_.end()) {
      return it->second;
    }
    unique_ptr<xml_document> document = make_unique<xml_document>();
    /* A file, or the document itself. */
    const xml_parse_result result = string_endswith(dictionary, ".xml") ?
                                        document->load_file(dictionary.c_str()) :
                                        document->load_buffer(dictionary.c_str(),
                                                              dictionary.size());
    int index = -1;
    if (result) {
      index = int(documents_.size());
      documents_.push_back(std::move(document));
    }
    document_map_[dictionary] = index;
    return index;
  }

  int find(const std::pair<int, std::string> &query, const xml_node &from)
  {
    const auto it = queries_.find(query);
    if (it != queries_.end()) {
      return it->second;
    }
    int first = 0;
    try {
      int last = -1;
      for (const PUGIXML_NAMESPACE::xpath_node &match : from.select_nodes(query.second.c_str())) {
        Node node;
        node.node = match.node();
        nodes_.push_back(node);
        const int id = int(nodes_.size()) - 1;
        if (last < 0) {
          first = id;
        }
        else {
          nodes_[last].next = id;
        }
        last = id;
      }
    }
    catch (const PUGIXML_NAMESPACE::xpath_exception &) {
      first = 0;
    }
    queries_[query] = first;
    return first;
  }

  vector<unique_ptr<xml_document>> documents_;
  map<std::string, int> document_map_;
  vector<Node> nodes_;
  /* Results by the node the query is relative to, or the negative document, and the query. */
  map<std::pair<int, std::string>, int> queries_;
};
#  endif

string metal_osl_camera_program_key(const string &bytecode_hash,
                                    const OSLCameraParams &params,
                                    const string &colorspace)
{
  MD5Hash md5;
  md5.append(bytecode_hash);
  md5.append(colorspace);
  for (const auto &it : params) {
    OSLCameraTranslateParam param;
    if (!osl_camera_param_convert(it.second.second, param)) {
      continue;
    }
    md5.append(it.first.string());
    md5.append(string_printf(":%d:%d:", int(param.type), param.size));
    if (param.type == OSLCameraTranslateParam::STRING) {
      md5.append(string((const char *)it.second.first.data()));
    }
  }
  return md5.get_hex();
}

bool metal_osl_camera_translate(const string &bytecode,
                                const OSLCameraParams &params,
                                const string &colorspace,
                                MetalOSLCameraProgram &program,
                                string &error)
{
  OSLCameraTranslateOptions options;
  options.colorspace = colorspace;
  static_assert(sizeof(cie_color_match) == sizeof(float) * 81 * 3, "Unexpected CIE table size");
  options.cie_color_match = &cie_color_match[0][0];
  /* The hash() of a string is the hash of the string type of the OSL runtime. */
  options.string_hash = [](const std::string &value) { return int(ustring(value).hash()); };
  for (const auto &it : params) {
    OSLCameraTranslateParam param;
    if (!osl_camera_param_convert(it.second.second, param)) {
      continue;
    }
    if (param.type == OSLCameraTranslateParam::STRING) {
      param.string_value = (const char *)it.second.first.data();
    }
    options.params[it.first.string()] = param;
  }

#  ifdef WITH_PUGIXML
  OSLCameraXMLDictionary dictionary;
  options.dictionary = &dictionary;
#  endif

  OSLCameraTranslateResult result;
  if (!osl_camera_translate_msl(bytecode, options, result, error)) {
    return false;
  }

  program.source = result.source;
  program.params.clear();
  for (const OSLCameraTranslateResult::Slot &slot : result.slots) {
    program.params.push_back({ustring(slot.name), slot.size, slot.offset, slot.is_int});
  }
  program.images.clear();
  for (const OSLCameraTranslateResult::Image &image : result.images) {
    program.images.push_back({image.filename, image.offset});
  }
  program.num_param_words = result.num_words;
  return true;
}

void metal_osl_camera_pack_params(const MetalOSLCameraProgram &program,
                                  const OSLCameraParams &params,
                                  const vector<int> &image_ids,
                                  vector<uint> &words)
{
  words.clear();
  words.resize(program.num_param_words, 0);
  for (size_t i = 0; i < program.images.size(); i++) {
    const int offset = program.images[i].offset;
    if (offset >= 0 && offset < program.num_param_words) {
      /* -1 is no image. */
      words[offset] = uint((i < image_ids.size()) ? image_ids[i] : -1);
    }
  }
  for (const MetalOSLCameraProgram::Param &param : program.params) {
    const auto it = params.find(param.name);
    if (it == params.end()) {
      continue;
    }
    const vector<uint8_t> &data = it->second.first;
    const size_t size = sizeof(uint) * param.size;
    if (data.size() >= size && param.offset + param.size <= program.num_param_words) {
      memcpy(&words[param.offset], data.data(), size);
    }
  }
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
