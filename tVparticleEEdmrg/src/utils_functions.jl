"""
Constructs a V_array with nVals elements linearly spaced between Vstart and Vend.
The array is made up of two subarrays containing positive and non-positive values
ordered with the smallest absolute values first.
"""
function lin_V_range(Vstart::Float64,Vend::Float64,nVals::Int64) 
    v_neg_reverse = -2
    #v_pos_reverse = +2 # for the positive range, keep the direction
    
    if Vstart <= 0
        if Vend > 0
            v_neg = collect(reverse(Vstart:abs(Vend-Vstart)/(nVals-1):0.0))
            v_pos = collect((Vend-Vstart)/(nVals-1):abs(Vend-Vstart)/(nVals-1):Vend)
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            # v_pos_LL = v_pos[v_pos.<v_pos_reverse]
            # v_pos_inf = reverse(v_pos[v_pos.>=v_pos_reverse])
            # return [v_neg_inf,v_neg_LL,v_pos_LL,v_pos_inf]
            return [v_neg_inf,v_neg_LL,v_pos]
        else
            v_neg = collect(reverse(Vstart:abs(Vend-Vstart)/(nVals-1):Vend))
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [ v_neg_inf, v_neg_LL]
        end 
    end

    v_pos = collect(Vstart:abs(Vend-Vstart)/(nVals-1):Vend)
    # v_pos_LL = v_pos[v_pos.<v_pos_reverse]
    # v_pos_inf = reverse(v_pos[v_pos.>=v_pos_reverse])
    # return [v_pos_LL,v_pos_inf]
    return [v_pos]

end


"""
Constructs a V_array with nVals elements logarithmically spaced between Vstart and Vend.
The array is made up of two subarrays containing positive and non-positive values
ordered with the smallest absolute values first.
"""
function log_V_range(Vstart::Float64,Vend::Float64,nVals::Int64)
    v_neg_reverse = -2
    #v_pos_reverse = +2 # for the positive range, keep the direction

    if Vstart <= 0
        if Vend > 0
            v_neg = [-1.0*10^v for v in log(10,1e-1):abs(log(10,1e-1)-log(10,abs(Vstart)))/((nVals-1)/2):log(10,abs(Vstart))] 
            v_pos = [10^v for v in log(10,1e-1):abs(log(10,1e-1)-log(10,Vend))/((nVals-1)/2):log(10,Vend)] 
            
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            # v_pos_LL = v_pos[v_pos.<v_pos_reverse]
            # v_pos_inf = reverse(v_pos[v_pos.>=v_pos_reverse])
            # return [v_neg_inf,v_neg_LL,v_pos_LL,v_pos_inf]
            return [v_neg_inf,v_neg_LL,v_pos]
        else
            v_neg = [-1.0*10^v for v in  log(10,abs(Vend)):abs(log(10,abs(Vend))-log(10,abs(Vstart)))/(nVals-1):log(10,abs(Vstart))]
            v_neg_inf = reverse(v_neg[v_neg.<v_neg_reverse])
            v_neg_LL = v_neg[v_neg.>=v_neg_reverse]
            return [ v_neg_inf, v_neg_LL]
        end 
    end

    v_pos = [10^v for v in log(10,Vstart):abs(log(10,Vend)-log(10,Vstart))/(nVals-1):log(10,Vend)]
    # v_pos_LL = v_pos[v_pos.<v_pos_reverse]
    # v_pos_inf = reverse(v_pos[v_pos.>=v_pos_reverse])

    # return [v_pos_LL,v_pos_inf]
    return [v_pos]
end
 