/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_shader_util.hh"
#include "node_util.hh"
#include "RNA_access.hh"
#include "UI_interface_layout.hh"
#include "UI_resources.hh"

namespace blender {
namespace nodes::node_shader_bsdf_diffraction_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>("Color"_ustr).default_value({1, 1, 1, 1});
  b.add_input<decl::Vector>("Normal"_ustr).hide_value();
  b.add_input<decl::Vector>("Tangent"_ustr)
      .hide_value()
      .description("Direction across the grooves in the surface tangent plane");
  b.add_output<decl::Shader>("BSDF"_ustr);
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->storage = MEM_new<NodeShaderDiffraction>(__func__);
  static_cast<NodeShaderDiffraction *>(node->storage)->quality = 1;
}

static void node_buttons(ui::Layout &layout, bContext * /*context*/, PointerRNA *ptr)
{
  layout.prop(ptr, "quality", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  ui::Layout &table_row = layout.row(true);
  table_row.prop(ptr, "optical_constants", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  table_row.op("node.diffraction_table_update", "", ICON_FILE_REFRESH);
  const bool has_table = RNA_pointer_get(ptr, "optical_constants").data != nullptr;
  for (const char *property : {"pitch", "depth", "duty_cycle", "incident_ior",
                               "ridge_ior", "ridge_extinction", "groove_ior",
                               "substrate_ior", "substrate_extinction"}) {
    ui::Layout &row = layout.row(false);
    if (has_table && (STREQ(property, "ridge_ior") || STREQ(property, "ridge_extinction") ||
                      STREQ(property, "substrate_ior") || STREQ(property, "substrate_extinction"))) {
      row.active_set(false);
    }
    row.prop(ptr, property, UI_ITEM_NONE, std::nullopt, ICON_NONE);
  }
}

}  // namespace nodes::node_shader_bsdf_diffraction_cc

void register_node_type_sh_bsdf_diffraction()
{
  namespace file_ns = nodes::node_shader_bsdf_diffraction_cc;
  static bke::bNodeType ntype;
  sh_node_type_base(&ntype, "ShaderNodeBsdfDiffraction"_ustr, SH_NODE_BSDF_DIFFRACTION);
  ntype.ui_name = "Diffraction BSDF";
  ntype.ui_description = "Spectral reflection and transmission from a smooth periodic relief grating";
  ntype.enum_name_legacy = "BSDF_DIFFRACTION";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.add_ui_poll = object_cycles_shader_nodes_poll;
  ntype.declare = file_ns::node_declare;
  ntype.initfunc = file_ns::node_init;
  ntype.draw_buttons = file_ns::node_buttons;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  bke::node_type_storage(
      ntype, "NodeShaderDiffraction", node_free_standard_storage, node_copy_standard_storage);
  bke::node_register_type(ntype);
}
}  // namespace blender
