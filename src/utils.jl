## util function to save the spectrum into a hdf5 file
function saveSpec(name::String, vals, vecs, dw)
    name == "" && return
    h5open(name*".h5", "w") do file    
        file["vals"] = vals
        file["vecs"] = vecs
        file["dw"] = real.(dw)
    end
end

## util function to read the spectrum from a hdf5 into a DataFrame
function readSpec(name::String)
    name == "" && return
    h5open(name*".h5", "r") do file
        vals = read(file["vals"])
        vecs = read(file["vecs"])
        dw   = read(file["dw"])
        return vals,vecs,dw
    end
end

## save sparse matrices
function saveMatrix(name::String, H, R, S)
    name == "" && return
    h5open(name*".h5", "cw") do file    
        (nh,h) = last(enumerate(H))
        (nr,r) = last(enumerate(R))
        (ns,s) = last(enumerate(S))

        create_group(file, "H/$(floor(Int,nh))")
        create_group(file, "R/$(floor(Int,nr))")
        ns>0 && create_group(file, "S/$(floor(Int,ns))")

        Hnz = findnz(h)
        Rnz = findnz(r)
        (ns>0) && (Snz = findnz(s))

        file["H/$(floor(Int,nh))/row"]    = Hnz[1]
        file["H/$(floor(Int,nh))/column"] = Hnz[2]
        file["H/$(floor(Int,nh))/val"]    = Hnz[3]

        file["R/$(floor(Int,nr))/row"]    = Rnz[1]
        file["R/$(floor(Int,nr))/column"] = Rnz[2]
        file["R/$(floor(Int,nr))/val"]    = Rnz[3]

        (ns>0) && (file["S/$(floor(Int,ns))/row"]    = Snz[1])
        (ns>0) && (file["S/$(floor(Int,ns))/column"] = Snz[2])
        (ns>0) && (file["S/$(floor(Int,ns))/val"]    = Snz[3])
    end
end

## read sparse matrix
function readMatrix(name::String)
    name == "" && return
    h5open(name*".h5", "r") do file    
        @show file 
        return map(1:6) do i
            row = file["/R/$(floor(Int,i))/row"]
            column = read(file, "R/$(floor(Int,i))/column")
            val = read(file, "R/$(floor(Int,i))/val")
            return sparse(row,column,val)
        end
    end
end

# util function to save data into a DataFrame
function readData(name::String)
    name == "" && return

    df = DataFrame()
    h5open(name*".h5", "r") do file
        for (name,group) in zip(keys(file),file)
            temp = Vector{typeof(read(group[string(1)]))}([])
            map(1:length(group)) do (i)
                append!(temp, [read(group[string(i)])])
            end
            df[!,name] = temp
        end 
    end
    return df
end

## returns unique elements present in a vector
function rmDoubles(vec)
    unique = []
    for v in vec
        !(v ∈ unique) && push!(unique, v)
    end
    return unique
end

## spinflips
function spinflip(spin::Tuple{Vararg{Int64}}, pos::Tuple{Vararg{Int64}})
    res =  map(enumerate(spin)) do (n,s)
        return n ∈ pos ? (s+1)%2 : s
    end
    return Tuple(res)
end

function sgn(x) 
    x>0 && return +1
    x<0 && return -1
    return 0
end

function find_inds(nInit, order::Int, V1)
    is = nInit
    curr_inds = nInit

    for o in 1:order
        next_inds = []
        for i in curr_inds
            inds, vals = findnz(V1[i,:])
            push!(next_inds, inds)
        end

        curr_inds = filter(x->!(x ∈ is), rmDoubles(vcat(next_inds...)))
        append!(is, curr_inds)
    end

    return rmDoubles(vcat(is...))
end
