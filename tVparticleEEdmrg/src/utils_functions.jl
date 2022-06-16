"""
Constructs a V_array with nVals elements linearly spaced between Vstart and Vend.
The array is made up of up to three subarrays containing positive, non-positive values
in LL phase, and negative values < -2. These are sorted to approch +2 and -2 starting
from 0 and -2 from the right starting from the largest negative value.
This ensures that states from privious interactions can be used as starting configurations
for next steps to help dmrg with convergence.
"""
function lin_range(Vstart::Float64, Vend::Float64, nVals::Int64)

    if Vstart == Vend || nVals == 1
        return [Vstart]
    end

    v_neg_reverse = -2

    if Vstart < 0
        if Vend >= 0
            v_neg = collect(reverse(Vstart:abs(Vend - Vstart)/(nVals-1):-abs(Vend - Vstart)/(nVals-1)))
            v_pos = collect(0.0:abs(Vend - Vstart)/(nVals-1):Vend)
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [v_neg_inf, v_neg_LL, v_pos]
        else
            v_neg = collect(reverse(Vstart:abs(Vend - Vstart)/(nVals-1):Vend))
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [v_neg_inf, v_neg_LL]
        end
    end

    v_pos = collect(Vstart:abs(Vend - Vstart)/(nVals-1):Vend)
    return [v_pos]

end


"""
Constructs a V_array with nVals elements logarithmically spaced between Vstart and Vend.
The array is made up of up to three subarrays containing positive, non-positive values
in LL phase, and negative values < -2. These are sorted to approch +2 and -2 starting
from 0 and -2 from the right starting from the largest negative value.
This ensures that states from privious interactions can be used as starting configurations
for next steps to help dmrg with convergence.
"""
function log_range(Vstart::Float64, Vend::Float64, nVals::Int64)

    if Vstart == Vend || nVals == 1
        return [Vstart]
    end

    v_neg_reverse = -2

    if Vstart < 0
        if Vend >= 0
            v_neg = [-1.0 * 10^v for v in log(10, 1e-1):abs(log(10, 1e-1) - log(10, abs(Vstart)))/((nVals-1)/2):log(10, abs(Vstart))]
            v_pos = [10^v for v in log(10, 1e-1):abs(log(10, 1e-1) - log(10, Vend))/((nVals-1)/2):log(10, Vend)]

            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [v_neg_inf, v_neg_LL, v_pos]
        else
            v_neg = [-1.0 * 10^v for v in log(10, abs(Vend)):abs(log(10, abs(Vend)) - log(10, abs(Vstart)))/(nVals-1):log(10, abs(Vstart))]
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [v_neg_inf, v_neg_LL]
        end
    end

    v_pos = [10^v for v in log(10, Vstart):abs(log(10, Vend) - log(10, Vstart))/(nVals-1):log(10, Vend)]

    return [v_pos]
end
