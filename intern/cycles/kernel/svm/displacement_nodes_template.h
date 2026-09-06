/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Shared whitelist and implementation: the host certifies every emitted opcode. */

DISPLACEMENT_SVM_NODE(NODE_END, return;)

DISPLACEMENT_SVM_NODE(NODE_VALUE_F,
                      svm_node_value_f<float>(stack, svm_node_get<SVMNodeValueF>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_VALUE_F_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_value_f<dual1>(stack, svm_node_get<SVMNodeValueF>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(NODE_VALUE_V,
                      svm_node_value_v<float3>(stack, svm_node_get<SVMNodeValueV>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_VALUE_V_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_value_v<dual3>(stack, svm_node_get<SVMNodeValueV>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(
    NODE_ATTR, svm_node_attr_surface(kg, sd, stack, svm_node_get<SVMNodeAttr>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_ATTR_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_attr_derivative(kg, sd, stack, svm_node_get<SVMNodeAttr>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(NODE_CONVERT,
                      svm_node_convert<float, float3>(kg,
                                                      stack,
                                                      svm_node_get<SVMNodeConvert>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_CONVERT_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_convert<dual1, dual3>(kg, stack, svm_node_get<SVMNodeConvert>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(
    NODE_TEX_IMAGE,
    svm_node_tex_image<float3>(kg, sd, stack, svm_node_get<SVMNodeTexImage>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_TEX_IMAGE_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_tex_image<dual3>(kg, sd, stack, svm_node_get<SVMNodeTexImage>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(NODE_TEX_NOISE,
                      svm_node_tex_noise(stack, svm_node_get<SVMNodeTexNoise>(kg, &offset));)

DISPLACEMENT_SVM_NODE(NODE_RGB_RAMP, {
  const ccl_global auto &node = svm_node_get<SVMNodeRGBRamp>(kg, &offset);
  offset = svm_node_rgb_ramp(kg, stack, node, offset);
})

DISPLACEMENT_SVM_NODE(NODE_DISPLACEMENT,
                      svm_node_displacement<node_feature_mask>(
                          kg, sd, stack, svm_node_get<SVMNodeDisplacement>(kg, &offset));)

DISPLACEMENT_SVM_NODE(NODE_SET_DISPLACEMENT,
                      svm_node_set_displacement<node_feature_mask>(
                          sd, stack, svm_node_get<SVMNodeSetDisplacement>(kg, &offset));)

DISPLACEMENT_SVM_NODE(NODE_MAPPING,
                      svm_node_mapping<float3>(stack, svm_node_get<SVMNodeMapping>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_MAPPING_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_mapping<dual3>(stack, svm_node_get<SVMNodeMapping>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(
    NODE_TEXTURE_MAPPING,
    svm_node_texture_mapping<float3>(stack, svm_node_get<SVMNodeTextureMapping>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_TEXTURE_MAPPING_DERIVATIVE, IF_NOT_KERNEL_NODES_FEATURE(VOLUME) {
      svm_node_texture_mapping<dual3>(stack, svm_node_get<SVMNodeTextureMapping>(kg, &offset));
    })

DISPLACEMENT_SVM_NODE(
    NODE_GEOMETRY,
    svm_node_geometry<float3>(kg, sd, stack, svm_node_get<SVMNodeGeometry>(kg, &offset));)

DISPLACEMENT_SVM_NODE(
    NODE_GEOMETRY_DERIVATIVE,
    svm_node_geometry<dual3>(kg, sd, stack, svm_node_get<SVMNodeGeometry>(kg, &offset));)
