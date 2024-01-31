include("../../src/PertTheory.jl")

include("./hardcoreBosons.jl")

# gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75]
gs = collect(-0.:-0.05:-3.0)
Ls = [(4,4),(6,6),(8,8)]
ns = [3,5]

for g in gs
    #parameters
    J = -1.;
    h = 0.;
    # g = gs[parse(Int, ARGS[1])]
    L = Ls[1]
    N = prod(L);

    n = 5

    single_dw_basis = map(1:n) do l
        s = fill(0,n)
        s[l] = 1
        return s
    end

	#initial state
    dw_basis = vec(collect(Iterators.product(fill(single_dw_basis,L[2])...)));
    @show size(dw_basis)
    init_idx = findall(x->x==Tuple(fill(single_dw_basis[div(n+1,2)], L[2])), dw_basis)[1]

	psi=zeros(length(dw_basis))
	psi[init_idx]=1

    # imb_precalc = map(dw_basis) do dw
    #     return imbalance(dw, L, n)
    # end

    basis_table = Dict(
        map(enumerate(dw_basis)) do (i, spin)
            return (spin, i)
        end
    );

	H = build_matrix(dw_basis, basis_table, (L,J,g,h))

    println("Starting calculation of eigenvectors/-values")
    @time "eigen: " vals, vecs = eigen(Matrix(H))

    df = DataFrame(vals = vals, groundstate = vecs[:,1])

    CSV.write("../../data/spec_Eff_Bound_hardcoreBosonsDiffEnergy_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g).csv", df);
end
