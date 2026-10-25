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

static bool node_is_grating(const bNode *node)
{
  /* Nodes without storage come from builds where Grating is the only model. */
  return node == nullptr || node->storage == nullptr ||
         static_cast<const NodeShaderDiffraction *>(node->storage)->quality ==
             SHD_DIFFRACTION_MODEL_GRATING;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNodeTree *ntree = b.tree_or_null();
  const bool is_gpu_internal = ntree && (ntree->flag & NTREE_IS_GPU_SHADER_INTERNAL);
  const bNode *node = b.node_or_null();
  const bool is_grating = node_is_grating(node);
  const bool is_rectangular = node ? node->custom1 == SHD_DIFFRACTION_RECTANGULAR : true;
  const bool is_2d = node ? node->custom2 == SHD_DIFFRACTION_GRATING_2D : false;

  b.add_input<decl::Color>("Color"_ustr).default_value({0.9f, 0.9f, 0.9f, 1.0f});
  b.add_input<decl::Float>("Roughness"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .available(is_grating)
      .description("Angular spread of each diffraction order, like Glossy roughness");

  /* Vector sockets: X = the first axis (the only one used in 1D), Y = the second axis, only
   * read when Grating Type is 2D. Z is unused. */
  b.add_input<decl::Vector>("Pitch"_ustr)
      .default_value({1.6f, 1.6f, 1.6f})
      .min(0.05f)
      .max(50.0f)
      .available(is_grating)
      .description(is_2d ? "Grating pitch per axis (X, Y), in micrometers. A CD track pitch is "
                           "1.6" :
                           "Grating pitch (groove spacing), in micrometers. A CD track pitch is "
                           "1.6. Only X is used");
  b.add_input<decl::Vector>("Groove Depth"_ustr)
      .default_value({110.0f, 110.0f, 110.0f})
      .min(0.0f)
      .max(2000.0f)
      .available(is_grating)
      .description(is_2d ? "Groove depth per axis (X, Y), in nanometers. Sets how energy splits "
                           "between orders" :
                           "Groove depth, in nanometers. Sets how energy splits between orders. "
                           "Only X is used");
  b.add_input<decl::Vector>("Duty Cycle"_ustr)
      .default_value({0.45f, 0.45f, 0.45f})
      .min(0.01f)
      .max(0.99f)
      .available(is_grating && is_rectangular)
      .description(is_2d ? "Pit-to-pitch width fraction per axis (X, Y)" :
                           "Pit-to-pitch width fraction. Only X is used");
  /* X rotates the first lattice axis, Y the second one (2D only), independently, so the two
   * do not have to stay perpendicular. Z is unused. */
  b.add_input<decl::Vector>("Rotation"_ustr)
      .default_value({0.0f, 0.25f, 0.0f})
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .available(is_grating)
      .description(is_2d ? "Rotation of each lattice axis from the tangent, X and Y, with 1.0 "
                           "going full circle. 0.25 between X and Y for a rectangular lattice" :
                           "Rotation of the grating direction from the tangent, with 1.0 going "
                           "full circle. Only X is used");
  b.add_input<decl::Vector>("Normal"_ustr).hide_value();
  b.add_input<decl::Vector>("Tangent"_ustr)
      .hide_value()
      .description(is_grating ? "Reference direction of the lattice in the surface tangent "
                                "plane, across the grooves at zero rotation. Defaults to the "
                                "tangent of the UV map" :
                                "Direction across the grooves in the surface tangent plane");
  b.add_input<decl::Float>("Weight"_ustr).available(is_gpu_internal);
  b.add_output<decl::Shader>("BSDF"_ustr);
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  NodeShaderDiffraction *storage = MEM_new<NodeShaderDiffraction>(__func__);
  storage->quality = SHD_DIFFRACTION_MODEL_GRATING;
  storage->flag = SHD_DIFFRACTION_FLAG_GRATING_DEFAULT;
  node->storage = storage;
  node->custom1 = SHD_DIFFRACTION_RECTANGULAR;
  node->custom2 = SHD_DIFFRACTION_GRATING_1D;
}

static void node_buttons(ui::Layout &layout, bContext * /*context*/, PointerRNA *ptr)
{
  layout.prop(ptr, "quality", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  if (node_is_grating(static_cast<const bNode *>(ptr->data))) {
    layout.prop(ptr, "grating_type", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
    layout.prop(ptr, "profile", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
    return;
  }
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

/* EEVEE has no spectral grating model, it previews the node as a glossy reflection tinted by its
 * color: with the roughness of the Grating model, and as a mirror for the physical models. */
static int node_shader_gpu_bsdf_diffraction(GPUMaterial *mat,
                                            bNode *node,
                                            bNodeExecData * /*execdata*/,
                                            GPUNodeStack *in,
                                            GPUNodeStack *out)
{
  /* Inputs: Color, Roughness, Pitch, Groove Depth, Duty Cycle, Rotation, Normal, Tangent. */
  if (!in[6].link) {
    GPU_link(mat, "world_normals_get", GPU_shading_data(), &in[6].link);
  }

  GPU_material_flag_set(mat, GPU_MATFLAG_GLOSSY);

  if (in[0].might_be_tinted()) {
    GPU_material_flag_set(mat, GPU_MATFLAG_REFLECTION_MAYBE_COLORED);
  }

  const float use_roughness = node_is_grating(node) ? 1.0f : 0.0f;
  return GPU_stack_link(mat,
                        node,
                        "node_bsdf_diffraction",
                        in,
                        out,
                        GPU_constant(&use_roughness),
                        GPU_kernel_globals(),
                        GPU_shading_data());
}

}  // namespace nodes::node_shader_bsdf_diffraction_cc

void register_node_type_sh_bsdf_diffraction()
{
  namespace file_ns = nodes::node_shader_bsdf_diffraction_cc;
  static bke::bNodeType ntype;
  sh_node_type_base(&ntype, "ShaderNodeBsdfDiffraction"_ustr, SH_NODE_BSDF_DIFFRACTION);
  ntype.ui_name = "Diffraction BSDF";
  ntype.ui_description =
      "Spectral reflection off a periodic grooved surface (a CD/DVD track, a holographic foil), "
      "splitting light into its diffraction orders";
  ntype.enum_name_legacy = "BSDF_DIFFRACTION";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.add_ui_poll = object_cycles_shader_nodes_poll;
  ntype.declare = file_ns::node_declare;
  ntype.initfunc = file_ns::node_init;
  ntype.draw_buttons = file_ns::node_buttons;
  ntype.gpu_fn = file_ns::node_shader_gpu_bsdf_diffraction;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  bke::node_type_storage(
      ntype, "NodeShaderDiffraction", node_free_standard_storage, node_copy_standard_storage);
  bke::node_register_type(ntype);
}
}  // namespace blender
