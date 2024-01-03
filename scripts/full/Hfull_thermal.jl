include("../src/SW.jl")
#using CUDA

# gs            = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-1.25,-1.5,-1.75,-2.0,-2.25,-2.5]
# beta_ranges   = [0.32:0.002:0.34,0.33:0.001:0.34,0.32:0.001:0.33,0.32:0.001:0.33,0.32:0.001:0.33,0.31:0.001:0.32,0.29:0.0001:0.3,0.17:0.001:0.18]
# lambda_ranges = [0.18:0.01:0.25,0.1:0.01:0.18,0.05:0.01:0.18,0.2:0.01:0.3,0.2:0.01:0.3,]

gs            = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-2.0]
beta_ranges   = [0.2:0.02:0.34,0.2:0.02:0.34,0.28:0.02:0.34,0.28:0.02:0.34,0.28:0.02:0.34,0.28:0.02:0.34]
lambda_ranges = [-1.:0.01:0.1,0.:0.01:0.1,0.:0.01:0.1,0.:0.01:0.1,0.:0.01:0.1,0.:0.01:0.1]

#for (g,betas,lambdas) in zip(gs,beta_ranges,lambda_ranges)
params = collect(zip(gs,beta_ranges,lambda_ranges))

let
  L= (4,4)
  N = prod(L);
  J = -1;
  n_order =1

  (g,betas,lambdas) = params[parse(Int, ARGS[1])]

  betas = 0.01:0.05:0.41
  lambdas = 0.:0.05:0.5

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

  twoCorrelM  = build_twoCorrel(spin_basis, spin_basis_table)
  xPolM       = build_xPol(spin_basis, spin_basis_table)


  betas = collect(betas)
  lambdas = collect(lambdas)
  data = []

  @show g
  for beta in betas
      @show beta

      res = vecs
      @time res *= diagm(exp.(-beta .* vals))
      @time res *= Transpose(vecs)

      for lambd in lambdas
          @time resLambd = res*diagm(exp.(-lambd .* diag(H0)))
          @time Z = tr(resLambd)
          @time H_mean = tr(H_full*resLambd)
          @time N_mean = tr(H0*resLambd)
          @time x_mean = tr(xPolM*resLambd)
          @time TC_mean = tr(twoCorrelM*resLambd)

          @show (beta, lambd, Z, H_mean, N_mean, H_mean/Z, N_mean/Z, x_mean/Z, TC_mean/Z)
          push!(data, [beta, lambd, H_mean, N_mean, Z, H_mean/Z, N_mean/Z, x_mean/Z, TC_mean/Z])
      end
  end

  df = DataFrame(
    beta      = [real(d[1]) for d in data], 
    lambd     = [real(d[2]) for d in data], 
    H         = [real(d[3]) for d in data], 
    N         = [real(d[4]) for d in data], 
    Z         = [real(d[5]) for d in data], 
    H_Z       = [real(d[6]) for d in data], 
    N_Z       = [real(d[7]) for d in data], 
    xPol      = [real(d[8]) for d in data], 
    twoCorrel = [real(d[9]) for d in data],
  )

  CSV.write("../data/thermGGE4_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
end
