# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Distillation."""
from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class Distillation(Module):
    """Distillation Base Class."""

    def __init__(self, student, teacher):
        super().__init__()
        self.student = ClassFactory.get_instance(ClassType.NETWORK, student)
        self.teacher = ClassFactory.get_instance(ClassType.NETWORK, teacher)
        self.teacher.freeze('teacher')

    def load_state_dict(self, state_dict=None, strict=True):
        """Load state dict."""
        self.teacher.exclude_weight_prefix = self.exclude_weight_prefix
        self.student.exclude_weight_prefix = self.exclude_weight_prefix
        if self.training:
            return self.teacher.load_state_dict(state_dict, strict)
        else:
            return self.student.load_state_dict(state_dict, strict)

    def state_dict(self):
        """Save state dict."""
        return self.student.state_dict()


@ClassFactory.register(ClassType.NETWORK)
class TinyBertDistil(Distillation):
    """TinyBertDistil for Classification task."""

    def __init__(self, student, teacher, header=None):
        super(TinyBertDistil, self).__init__(student, teacher)
        self.loss_mse = ops.MSELoss()
        self.head = ClassFactory.get_instance(ClassType.NETWORK, header)

    def call(self, input_ids, token_type_ids, attention_mask, **kwargs):
        """Call model."""
        pooled_output, sequence_output = self.student(input_ids, token_type_ids, attention_mask, is_student=True)
        student_atts, student_reps = sequence_output[1:], sequence_output
        if not self.training:
            return self.head(pooled_output)
        att_loss, rep_loss = 0., 0.
        # no grade save memory
        # with ops.no_grad():
        _, teacher_output = self.teacher(input_ids, token_type_ids, attention_mask)
        teacher_atts, teacher_reps = teacher_output[1:], teacher_output
        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
        teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = ops.where(student_att <= -1e2, ops.zeros_like(student_att).cuda(), student_att)
            teacher_att = ops.where(teacher_att <= -1e2, ops.zeros_like(teacher_att).cuda(), teacher_att)
            att_loss += self.loss_mse(student_att, teacher_att)

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            rep_loss += self.loss_mse(student_rep, teacher_rep)

        loss = att_loss + rep_loss
        return loss
