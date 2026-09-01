# Mantis v0.6.0 上游 bug 热修复（加载后按相同签名替换原方法）
#
# Bug: src/Forms/FormOperators/Wedge.jl 中「1-forms ∧ 1-forms in 3D」的全部
# 4 个变体，第 2、3 分量的循环体误写入 wedge_eval[1]（复制粘贴错误），导致
# α¹∧β¹ 的 dξ³∧dξ¹、dξ¹∧dξ² 分量错误/未初始化。质量矩阵与混合矩阵不受影响
# （只走 1∧2、2∧1 路径），但叉乘载荷 (J×H) 必须经过 1∧1。
# 已按原实现修正索引；约定与原注释一致（2-形式分量为循环序 proxy）：
#   α¹∧β¹ = (α₂β₃-α₃β₂)dξ²∧dξ³ + (α₃β₁-α₁β₃)dξ³∧dξ¹ + (α₁β₂-α₂β₁)dξ¹∧dξ²
# TODO: 向 MantisFEM/Mantis.jl 上游报告。

import Mantis.Forms: _evaluate_wedge, AbstractForm, evaluate
import Mantis.Points

# (rank 0 ∧ rank 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
)
    e1, _ = evaluate(form_expression_1, element_id, xi)
    e2, _ = evaluate(form_expression_2, element_id, xi)
    n = size(e1[1], 1)
    wedge_eval = [zeros(Float64, n, 1) for _ in 1:3]
    for ci in CartesianIndices(wedge_eval[1])
        wedge_eval[1][ci] = e1[2][ci] * e2[3][ci] - e1[3][ci] * e2[2][ci]
        wedge_eval[2][ci] = e1[3][ci] * e2[1][ci] - e1[1][ci] * e2[3][ci]
        wedge_eval[3][ci] = e1[1][ci] * e2[2][ci] - e1[2][ci] * e2[1][ci]
    end
    return wedge_eval, [[1]]
end

# (rank > 0 ∧ rank 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1}
    e1, idx1 = evaluate(form_expression_1, element_id, xi)
    e2, _ = evaluate(form_expression_2, element_id, xi)
    nb1 = size.(idx1, 1)
    n = size(e1[1], 1)
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(undef, n, nb1...) for _ in 1:3
    ]
    for ci in CartesianIndices(wedge_eval[1])
        (point, _) = Tuple(ci)
        wedge_eval[1][ci] = e1[2][ci] * e2[3][point, 1] - e1[3][ci] * e2[2][point, 1]
        wedge_eval[2][ci] = e1[3][ci] * e2[1][point, 1] - e1[1][ci] * e2[3][point, 1]
        wedge_eval[3][ci] = e1[1][ci] * e2[2][point, 1] - e1[2][ci] * e2[1][point, 1]
    end
    return wedge_eval, idx1
end

# (rank 0 ∧ rank > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_2}
    e1, _ = evaluate(form_expression_1, element_id, xi)
    e2, idx2 = evaluate(form_expression_2, element_id, xi)
    nb2 = size.(idx2, 1)
    n = size(e1[1], 1)
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(undef, n, nb2...) for _ in 1:3
    ]
    for ci in CartesianIndices(wedge_eval[1])
        (point, _) = Tuple(ci)
        wedge_eval[1][ci] = e1[2][point, 1] * e2[3][ci] - e1[3][point, 1] * e2[2][ci]
        wedge_eval[2][ci] = e1[3][point, 1] * e2[1][ci] - e1[1][point, 1] * e2[3][ci]
        wedge_eval[3][ci] = e1[1][point, 1] * e2[2][ci] - e1[2][point, 1] * e2[1][ci]
    end
    return wedge_eval, idx2
end

# (rank > 0 ∧ rank > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1, expression_rank_2}
    e1, idx1 = evaluate(form_expression_1, element_id, xi)
    e2, idx2 = evaluate(form_expression_2, element_id, xi)
    nb1 = size.(idx1, 1)
    nb2 = size.(idx2, 1)
    n = size(e1[1], 1)
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef, n, nb1..., nb2...
        ) for _ in 1:3
    ]
    for ci in CartesianIndices(wedge_eval[1])
        (point, b1, b2) = Tuple(ci)
        wedge_eval[1][ci] =
            e1[2][point, b1] * e2[3][point, b2] - e1[3][point, b1] * e2[2][point, b2]
        wedge_eval[2][ci] =
            e1[3][point, b1] * e2[1][point, b2] - e1[1][point, b1] * e2[3][point, b2]
        wedge_eval[3][ci] =
            e1[1][point, b1] * e2[2][point, b2] - e1[2][point, b1] * e2[1][point, b2]
    end
    return wedge_eval, vcat(idx1, idx2)
end
