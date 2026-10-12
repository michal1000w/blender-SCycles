/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_shader_util.hh"

#include "BKE_node_runtime.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

namespace blender::nodes::node_shader_volume_ior_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNodeTree *ntree = b.tree_or_null();
  const bool is_gpu_internal = ntree && (ntree->flag & NTREE_IS_GPU_SHADER_INTERNAL);

  b.add_input<decl::Float>("IOR"_ustr)
      .default_value(1.0f)
      .min(0.01f)
      .max(4.0f)
      .description(
          "Index of refraction of the volume. Rays bend where it varies in space: link a "
          "texture or an attribute, or name an attribute below which scales the difference "
          "from 1")
      .make_available([](bNode &node) { node.custom1 = SHD_VOLUME_IOR_DIRECT; });
  b.add_input<decl::Float>("Temperature"_ustr)
      .default_value(1000.0f)
      .min(0.0f)
      .max(10000.0f)
      .description(
          "Temperature of the gas in kelvin, multiplied by the attribute if one is set. The "
          "gas is never colder than the ambient temperature")
      .make_available([](bNode &node) { node.custom1 = SHD_VOLUME_IOR_TEMPERATURE; });
  b.add_input<decl::String>("Attribute"_ustr)
      .description(
          "Name of a volume grid or attribute to take the field from, such as the temperature "
          "or heat of a fluid simulation");
  b.add_input<decl::Float>("Ambient Temperature"_ustr)
      .default_value(293.15f)
      .min(1.0f)
      .max(10000.0f)
      .description("Temperature of the surrounding gas in kelvin")
      .make_available([](bNode &node) { node.custom1 = SHD_VOLUME_IOR_TEMPERATURE; });
  b.add_input<decl::Float>("Pressure"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .max(1000.0f)
      .description("Pressure of the gas in atmospheres")
      .make_available([](bNode &node) { node.custom1 = SHD_VOLUME_IOR_TEMPERATURE; });
  b.add_input<decl::Float>("Gas IOR"_ustr)
      .default_value(1.000278f)
      .min(1.0f)
      .max(2.0f)
      .description(
          "Index of refraction of the gas at 15 \u00b0C and one atmosphere: 1.000278 for air, "
          "1.000450 for carbon dioxide, 1.000036 for helium")
      .make_available([](bNode &node) { node.custom1 = SHD_VOLUME_IOR_TEMPERATURE; });
  b.add_input<decl::Float>("Strength"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .max(100000.0f)
      .description(
          "Scale of the difference of the index of refraction from 1. Heat haze is subtle at "
          "the physically correct value of 1: higher values exaggerate it");
  b.add_input<decl::Float>("Weight"_ustr).available(is_gpu_internal);
  b.add_output<decl::Shader>("Volume"_ustr).translation_context(BLT_I18NCONTEXT_ID_ID);
}

static void node_shader_buts_volume_ior(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "mode", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
  layout.prop(ptr, "step_size", ui::ITEM_R_SPLIT_EMPTY_NAME, std::nullopt, ICON_NONE);
}

static void node_shader_init_volume_ior(bNodeTree * /*ntree*/, bNode *node)
{
  node->custom1 = SHD_VOLUME_IOR_TEMPERATURE;
  node->custom3 = 0.0f;
}

static void node_shader_update_volume_ior(bNodeTree *ntree, bNode *node)
{
  const int mode = node->custom1;
  for (bNodeSocket &sock : node->inputs) {
    if (STREQ(sock.name, "IOR")) {
      bke::node_set_socket_availability(*ntree, sock, mode == SHD_VOLUME_IOR_DIRECT);
    }
    else if (STR_ELEM(sock.name, "Temperature", "Ambient Temperature", "Pressure", "Gas IOR")) {
      bke::node_set_socket_availability(*ntree, sock, mode == SHD_VOLUME_IOR_TEMPERATURE);
    }
  }
}

static int node_shader_gpu_volume_ior(GPUMaterial *mat,
                                      bNode *node,
                                      bNodeExecData * /*execdata*/,
                                      GPUNodeStack *in,
                                      GPUNodeStack *out)
{
  return GPU_stack_link(
      mat, node, "node_volume_ior", in, out, GPU_kernel_globals(), GPU_shading_data());
}

}  // namespace blender::nodes::node_shader_volume_ior_cc

void blender::register_node_type_sh_volume_ior()
{
  namespace file_ns = nodes::node_shader_volume_ior_cc;

  static bke::bNodeType ntype;
  sh_node_type_base(&ntype, "ShaderNodeVolumeIOR"_ustr, SH_NODE_VOLUME_IOR);
  ntype.ui_name = "Volume IOR";
  ntype.ui_description =
      "Index of refraction of a volume, which bends light where it varies in space: heat haze "
      "above fire and hot surfaces, mirages, gases and liquids that mix";
  ntype.enum_name_legacy = "VOLUME_IOR";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.declare = file_ns::node_declare;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.draw_buttons = file_ns::node_shader_buts_volume_ior;
  ntype.initfunc = file_ns::node_shader_init_volume_ior;
  ntype.gpu_fn = file_ns::node_shader_gpu_volume_ior;
  ntype.updatefunc = file_ns::node_shader_update_volume_ior;

  bke::node_register_type(ntype);
}
