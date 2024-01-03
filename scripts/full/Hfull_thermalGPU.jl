include("../src/SW.jl")
using CUDA

gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-2.0]

#map(gs) do g
let
  L= (4,4)
  N = prod(L);
  J = -1;
  n_order =1

  # ni = 8 #parse(Int, ARGS[1])
  g = gs[8]
  device!(6)

  h = -0.

  next_neighbours = nearest_neighbours(L, collect(1:prod(L)))

  spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
  dw_precalc = map(spin_basis) do spin
      return domainWallL(spin, L)
  end

  ### sort basis according to domain wall length ###
  sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
  dw_precalc  = [d[1] for d in sorted_spin_basis]
  spin_basis  = [d[2] for d in sorted_spin_basis]

  spin_basis_table = Dict(
      map(enumerate(zip(spin_basis,dw_precalc))) do (i, (spin, dw))
          return (spin, (i,dw))
      end
  );

  domainWall_table = Dict(
      map(enumerate(dw_precalc)) do (i,  dw)
          return (i, dw)
      end
  );

  ### initial state ###
  #init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
  #init_idx = first(table[Tuple(init_spin)]);

  #psi=zeros(length(spin_basis))
  #psi[init_idx]=1
  #psi = sparse(psi)

  ### build Hamiltonians ###
  H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
  H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
  V1 = H1+R1;

  #======================== full =============================#
  H_full = H0+V1

  #=
  #==================== perturbative =========================#
  vecH, vecR, vecS = SW_transformation(H0, H1, R1, domainWall_table, n_order)
  H_pert = dropzeros(reduce(+, vecH; init=H0))

  ### observables ###
  imb_precalc = map(s -> imbalance(s, L), spin_basis)
  H0_corr = build_corr(H0, vecS)
  D0      = dropzeros(reduce(+, vecH))
  =#

  vals,vecs,_ = readSpec("../data/spec_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

  betas = collect(0.1:0.01:0.3)

  @show g
  data = map(betas) do beta
      @show beta

      res = cu(vecs)

      @time res *= cu(diagm(exp.(-beta .* vals)))
      @time res *= cu(Transpose(vecs))
      @time Z = tr(res)
      @time H_mean = tr(cu(H_full)*res)
      @time N_mean = tr(cu(H0)*res)

      @show (beta, Z, H_mean, N_mean, H_mean/Z, N_mean/Z)
      return [beta, H_mean, N_mean, Z, H_mean/Z, N_mean/Z]
  end

  df = DataFrame(beta = [real(d[1]) for d in data], H = [real(d[2]) for d in data], N = [real(d[3]) for d in data], Z = [real(d[4]) for d in data], H_Z = [real(d[5]) for d in data], N_Z = [real(d[6]) for d in data])

  CSV.write("../data/therm_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
  return
end
