/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_shader_util.hh"

#include "BLI_string.hh"

#include "BKE_node_runtime.hh"

#include "IMB_colormanagement.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

namespace blender::nodes::node_shader_volume_fast_cc {

enum SocketId {
  SOCK_SCATTER_COLOR = 0,
  SOCK_COLOR_ATTRIBUTE,
  SOCK_DENSITY,
  SOCK_DENSITY_ATTRIBUTE,
  SOCK_DENSITY_CUTOFF,
  SOCK_SCATTER_STRENGTH,
  SOCK_SCATTER_ATTRIBUTE,
  SOCK_ANISOTROPY,
  SOCK_IOR,
  SOCK_BACKSCATTER,
  SOCK_ALPHA,
  SOCK_DIAMETER,
  SOCK_ABSORPTION_COLOR,
  SOCK_ABSORPTION_STRENGTH,
  SOCK_EMISSION_STRENGTH,
  SOCK_EMISSION_COLOR,
  SOCK_EMISSION_ATTRIBUTE,
  SOCK_FLAME_CUTOFF,
  SOCK_BLACKBODY_STRENGTH,
  SOCK_BLACKBODY_TINT,
  SOCK_TEMPERATURE,
  SOCK_TEMPERATURE_ATTRIBUTE,
  SOCK_WEIGHT,
};

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNodeTree *ntree = b.tree_or_null();
  const bool is_gpu_internal = ntree && (ntree->flag & NTREE_IS_GPU_SHADER_INTERNAL);

  b.add_input<decl::Color>("Scatter Color"_ustr).default_value({0.5f, 0.5f, 0.5f, 1.0f});
  b.add_input<decl::String>("Color Attribute"_ustr);
  b.add_input<decl::Float>("Density"_ustr).default_value(1.0f).min(0.0f).max(1000.0f);
  b.add_input<decl::String>("Density Attribute"_ustr).default_value("density");
  b.add_input<decl::Float>("Density Cutoff"_ustr)
      .default_value(0.001f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR);
  b.add_input<decl::Float>("Scatter Strength"_ustr).default_value(1.0f).min(0.0f).max(1000.0f);
  b.add_input<decl::String>("Scatter Attribute"_ustr);
  b.add_input<decl::Float>("Anisotropy"_ustr)
      .default_value(0.0f)
      .min(-1.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .description(
          "Directionality of scattered light: negative is backward and positive is forward")
      .make_available([](bNode &node) { node.custom1 = SHD_PHASE_HENYEY_GREENSTEIN; });
  b.add_input<decl::Float>("IOR"_ustr)
      .default_value(1.33f)
      .min(1.0f)
      .max(2.0f)
      .description("Index of refraction used by the Fournier-Forand phase function")
      .make_available([](bNode &node) { node.custom1 = SHD_PHASE_FOURNIER_FORAND; });
  b.add_input<decl::Float>("Backscatter"_ustr)
      .default_value(0.1f)
      .min(0.0f)
      .max(0.5f)
      .description("Fraction of light scattered backwards by Fournier-Forand")
      .make_available([](bNode &node) { node.custom1 = SHD_PHASE_FOURNIER_FORAND; });
  b.add_input<decl::Float>("Alpha"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(500.0f)
      .description("Draine phase-function shape parameter")
      .make_available([](bNode &node) { node.custom1 = SHD_PHASE_DRAINE; });
  b.add_input<decl::Float>("Diameter"_ustr)
      .default_value(20.0f)
      .min(0.0f)
      .max(50.0f)
      .description("Particle diameter in micrometers for the Mie approximation")
      .make_available([](bNode &node) { node.custom1 = SHD_PHASE_MIE; });
  b.add_input<decl::Color>("Absorption Color"_ustr).default_value({0.0f, 0.0f, 0.0f, 1.0f});
  b.add_input<decl::Float>("Absorption Strength"_ustr).default_value(0.0f).min(0.0f).max(1000.0f);
  b.add_input<decl::Float>("Emission Strength"_ustr).default_value(0.0f).min(0.0f).max(1000.0f);
  b.add_input<decl::Color>("Emission Color"_ustr).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::String>("Emission Attribute"_ustr).default_value("flames");
  b.add_input<decl::Float>("Flame Cutoff"_ustr)
      .default_value(0.001f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .description("Ignore weaker flame values for cleaner and faster fire emission");
  b.add_input<decl::Float>("Blackbody Strength"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(1000.0f)
      .description(
          "Linear multiplier for physically based blackbody emission; values above one increase "
          "brightness");
  b.add_input<decl::Color>("Blackbody Tint"_ustr).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Float>("Temperature"_ustr)
      .default_value(1000.0f)
      .min(0.0f)
      .max(6500.0f)
      .subtype(PROP_COLOR_TEMPERATURE);
  b.add_input<decl::String>("Temperature Attribute"_ustr).default_value("temperature");
  b.add_input<decl::Float>("Weight"_ustr).available(is_gpu_internal);
  b.add_output<decl::Shader>("Volume"_ustr).translation_context(BLT_I18NCONTEXT_ID_ID);
}

static void node_shader_buts_fast_volume(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "phase", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
}

static void node_shader_init_fast_volume(bNodeTree * /*ntree*/, bNode *node)
{
  node->custom1 = SHD_PHASE_HENYEY_GREENSTEIN;
}

static void node_shader_update_fast_volume(bNodeTree *ntree, bNode *node)
{
  const int phase = node->custom1;
  for (bNodeSocket &sock : node->inputs) {
    if (STR_ELEM(sock.name, "IOR", "Backscatter")) {
      bke::node_set_socket_availability(*ntree, sock, phase == SHD_PHASE_FOURNIER_FORAND);
    }
    else if (STREQ(sock.name, "Anisotropy")) {
      bke::node_set_socket_availability(
          *ntree, sock, ELEM(phase, SHD_PHASE_HENYEY_GREENSTEIN, SHD_PHASE_DRAINE));
    }
    else if (STREQ(sock.name, "Alpha")) {
      bke::node_set_socket_availability(*ntree, sock, phase == SHD_PHASE_DRAINE);
    }
    else if (STREQ(sock.name, "Diameter")) {
      bke::node_set_socket_availability(*ntree, sock, phase == SHD_PHASE_MIE);
    }
  }
}

static void attribute_post_process(GPUMaterial *mat,
                                   const char *attribute_name,
                                   GPUNodeLink **attribute_link)
{
  if (STREQ(attribute_name, "color")) {
    GPU_link(mat, "node_attribute_color", *attribute_link, attribute_link);
  }
  else if (STREQ(attribute_name, "temperature")) {
    GPU_link(mat, "node_attribute_temperature", *attribute_link, attribute_link);
  }
}

static GPUNodeLink *volume_attribute(GPUMaterial *mat, const char *name, const bool post_process)
{
  if (name[0] == '\0') {
    return nullptr;
  }
  GPUNodeLink *link = GPU_attribute_with_default(mat, CD_AUTO_FROM_NAME, name, GPU_DEFAULT_1);
  if (post_process) {
    attribute_post_process(mat, name, &link);
  }
  return link;
}

static int node_shader_gpu_volume_fast(GPUMaterial *mat,
                                       bNode *node,
                                       bNodeExecData * /*execdata*/,
                                       GPUNodeStack *in,
                                       GPUNodeStack *out)
{
  const bool use_scatter = in[SOCK_DENSITY].socket_not_zero() &&
                           in[SOCK_SCATTER_STRENGTH].socket_not_zero() &&
                           in[SOCK_SCATTER_COLOR].socket_not_black();
  const bool use_absorption = in[SOCK_DENSITY].socket_not_zero() &&
                              in[SOCK_ABSORPTION_STRENGTH].socket_not_zero() &&
                              in[SOCK_ABSORPTION_COLOR].socket_not_black();
  const bool use_emission = in[SOCK_EMISSION_STRENGTH].socket_not_zero();
  const bool use_blackbody = in[SOCK_BLACKBODY_STRENGTH].socket_not_zero();

  if (use_scatter) {
    GPU_material_flag_set(mat, GPU_MATFLAG_VOLUME_SCATTER | GPU_MATFLAG_VOLUME_ABSORPTION);
  }
  if (use_absorption) {
    GPU_material_flag_set(mat, GPU_MATFLAG_VOLUME_ABSORPTION);
  }

  GPUNodeLink *density = nullptr;
  GPUNodeLink *color = nullptr;
  GPUNodeLink *scatter = nullptr;
  GPUNodeLink *emission = nullptr;
  GPUNodeLink *temperature = nullptr;

  for (bNodeSocket &sock : node->inputs) {
    if (sock.typeinfo->type != SOCK_STRING) {
      continue;
    }
    const auto *value = static_cast<const bNodeSocketValueString *>(sock.default_value);
    if (STREQ(sock.name, "Density Attribute")) {
      density = volume_attribute(mat, value->value, false);
    }
    else if (STREQ(sock.name, "Color Attribute")) {
      color = volume_attribute(mat, value->value, true);
    }
    else if (use_scatter && STREQ(sock.name, "Scatter Attribute")) {
      scatter = volume_attribute(mat, value->value, false);
    }
    else if ((use_emission || use_blackbody) && STREQ(sock.name, "Emission Attribute")) {
      emission = volume_attribute(mat, value->value, false);
    }
    else if (use_blackbody && STREQ(sock.name, "Temperature Attribute")) {
      temperature = volume_attribute(mat, value->value, true);
    }
  }

  static const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  density = density ? density : GPU_constant(white);
  color = color ? color : GPU_constant(white);
  scatter = scatter ? scatter : GPU_constant(white);
  emission = emission ? emission : GPU_constant(white);
  temperature = temperature ? temperature : GPU_constant(white);

  const int size = CM_TABLE + 1;
  float *data;
  float layer;
  if (use_blackbody) {
    data = MEM_new_array_uninitialized<float>(size * 4, "fast volume blackbody texture");
    IMB_colormanagement_blackbody_temperature_to_rgb_table(data, size, 800.0f, 12000.0f);
  }
  else {
    data = MEM_new_array_zeroed<float>(size * 4, "fast volume blackbody black");
  }
  GPUNodeLink *spectrummap = GPU_color_band(mat, size, data, &layer);

  return GPU_stack_link(mat,
                        node,
                        "node_volume_fast",
                        in,
                        out,
                        density,
                        color,
                        scatter,
                        emission,
                        temperature,
                        spectrummap,
                        GPU_constant(&layer));
}

}  // namespace blender::nodes::node_shader_volume_fast_cc

void blender::register_node_type_sh_volume_fast()
{
  namespace file_ns = nodes::node_shader_volume_fast_cc;

  static bke::bNodeType ntype;
  sh_node_type_base(&ntype, "ShaderNodeVolumeFast"_ustr, SH_NODE_VOLUME_FAST);
  ntype.ui_name = "Fast Volume";
  ntype.ui_description =
      "Efficient sparse volume shader with direct density, scatter, flame, and temperature fields";
  ntype.enum_name_legacy = "FAST_VOLUME";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.declare = file_ns::node_declare;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  ntype.default_width = bke::NodeWidth::_240;
  ntype.draw_buttons = file_ns::node_shader_buts_fast_volume;
  ntype.initfunc = file_ns::node_shader_init_fast_volume;
  ntype.gpu_fn = file_ns::node_shader_gpu_volume_fast;
  ntype.updatefunc = file_ns::node_shader_update_fast_volume;

  bke::node_register_type(ntype);
}
