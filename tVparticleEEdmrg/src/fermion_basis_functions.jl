using Printf


"""basis, nBasis = get_int_fermion_basis(n::Int64,L::Int64) -> Vector{Int64}, Int
return the integer fermion basis 'basis' with n fermions on L sites and the
number of basis elements 'nBasis'.
"""
function get_int_fermion_basis(n::Int64,L::Int64)
    L >= 1 || throw(DomainError(L, "At least 1 site is required."))
    L <= 64 || throw(DomainError(L, "Int64 basis does not support more than 64 sites."))
    n >= 0 || throw(DomainError(n, "At least 0 particles are required.")) 

    # Basis size.
    nBasis = binomial(L, n)
    v =2^n-1
    vectors = Vector{Int}(undef, nBasis)

    vectors[1] = v
    for i in 2:nBasis
        if CheckSite(v,1)==1
            j=findfirstEmpty(v)
            v=EmptySite(v,j-1)
            v=OccupySite(v,j)
        else
            j=findfirstOccupide(v)
            k=findFrsEmpAfterFrsOcc(v)
            v=EmptySite(v,k-j)
            v=OccupySite(v,k)
            for l in 1:(k-j-1)
                v=OccupySite(v,l)
            end
            for l in (k-j+1):(k-1)
                v=EmptySite(v,l)
            end
        end
        vectors[i] = v
    end

    return vectors, nBasis
end

""" find first bit=0 from the extreme right bit """
function findfirstEmpty(v::Int)
    return trailing_ones(v)+1
end

""" find first bit=1 from the extreme right bit """
function findfirstOccupide(v::Int)
    return trailing_zeros(v)+1
end


""" find first bit=0 after the first bit=1 from the extreme right bit"""
function findFrsEmpAfterFrsOcc(v::Int)
    return  findfirstOccupide(v)+trailing_ones(v>>>(findfirstOccupide(v)))+1 
end

""" set a bit to 1  """
function OccupySite(v::Int,bit::Int64)
    return ((1 << (bit-1)) | v)::Int64
end

""" set a bit to 0 """
function EmptySite(v::Int,bit::Int64)
    return convert(Int64, (~(1 << (bit-1))) & Int64(v))
end

""" read a bit """
function CheckSite(v::Int,bit::Int64)
    return Int64((v >> (bit-1)) & 1)
end


"""ni = convert_basis_vector(bi::Int64,L::Int64) -> Vector{Int64} 
Transforms the element bi from the integer basis into the
particle position basis i1,i2,i3,... such that i1<i2<i3<.... 
"""
function convert_basis_vector(bi::Int64)
    n_ones = count_ones(bi)
    ni = zeros(Int64,n_ones)

    pos = 1
    i = 1
    while n_ones > 0
        bi = bi >> 1
        n_ones_step = count_ones(bi)
        if n_ones_step < n_ones
            n_ones = n_ones_step
            ni[i] = pos
            i += 1
        end
        pos+=1
    end
    return ni
end

"""bi = convert_basis_vector(ni::Vector{Int64}) -> Int64
Transforms the vector ni of the particle position basis 
into the integer bi from the integer basis. It returns 
zero if the entries in ni are not compatible with the 
fermion basis, e.g. 1,1,2,3, or if ni are not ordered
in acending order
"""
function convert_basis_vector(ni::Vector{Int64})
    
    bi = 0
    i_last = ni[1]
    for i in ni
        if i_last > i
            return 0
        end
        bi = bi | (1 << (i-1))
        i_last = i
    end
    if count_ones(bi) != length(ni)
        return 0
    end
    return bi
end

"""index = get_position_int_basis(bi::Int64) -> Int64
Determines the position of the integer bi in the
integer fermion basis. <-- inefficent function!!, can be improved 
significantly but should be fast enough for n=4, N=16
"""
function get_position_int_basis(bi::Int64,n::Int64,L::Int64)
    # Basis size.
    nBasis = binomial(L, n)
    v =2^n-1 

    for i in 2:nBasis
        if v == bi
            return i - 1 
        end
        if CheckSite(v,1)==1
            j=findfirstEmpty(v)
            v=EmptySite(v,j-1)
            v=OccupySite(v,j)
        else
            j=findfirstOccupide(v)
            k=findFrsEmpAfterFrsOcc(v)
            v=EmptySite(v,k-j)
            v=OccupySite(v,k)
            for l in 1:(k-j-1)
                v=OccupySite(v,l)
            end
            for l in (k-j+1):(k-1)
                v=EmptySite(v,l)
            end
        end 
    end

    if v == bi
        return nBasis
    end
    error(@sprintf "Could not locate bi=%d in integer fermion basis of n=%d particles on L=%d sites." bi n L)
end

function get_position_int_basis(bi::Int64,int_basis::Vector{Int64})
    return searchsortedfirst(int_basis,bi)
end

"""index = get_position_int_basis(ni::Vector{Int64}) -> Int64
Determines the position of the vector ni from the
particle position basis in the corresponding integer 
fermion basis. It returns zero if the entries in ni 
are not compatible with the fermion basis, e.g. 1,1,2,3 
"""
function get_position_int_basis(ni::Vector{Int64},n::Int64,L::Int64)
    bi = translate_basis_vector(ni)
    if bi == 0
        return 0
    end
    return get_position_int_basis(bi,n,L)
end

function get_position_int_basis(ni::Vector{Int64},int_basis::Vector{Int64})
    bi = translate_basis_vector(ni)
    if bi == 0
        return 0
    end
    return get_position_int_basis(bi,int_basis)
end