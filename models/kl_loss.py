import torch
import torch.nn as nn
import torch.nn.functional as F


def get_kl_loss(teacher_logits, student_logits, student_labels, teacher_labels, temperature, distill_topk=None):

    # make sure the teacher_logits and student_logits have the same shape
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    # loss_fct = nn.KLDivLoss(reduction="sum")
    _, _, vocab_size = student_logits.shape

    # only compute loss in the completion part, not prompt
    student_mask = (student_labels != -100).unsqueeze(-1).expand_as(student_logits)  # batch_size, num_tokens, vocab_size
    student_logits_selected = torch.masked_select(student_logits, student_mask).view(-1, vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits, teacher_mask).view(-1, vocab_size)

    if distill_topk is not None:
        _, topk_teacher_indices = torch.topk(teacher_logits_selected, k=distill_topk, dim=-1)

        teacher_logits_selected = torch.gather(teacher_logits_selected, 1, topk_teacher_indices)
        student_logits_selected = torch.gather(student_logits_selected, 1, topk_teacher_indices)

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(teacher_logits_selected / temperature, dim=-1),
    ) * (temperature ** 2)
    # kl_loss = kl_loss / student_logits_selected.size(0)  # average per valid token

    return kl_loss