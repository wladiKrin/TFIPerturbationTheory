include("../src/SW.jl")

gs = [-0.3] #,-0.2,-0.3,-0.4,-0.5,-0.7] #
# ns = [1,2,3,4,5,6,7]
# params = collect(Iterators.product(ns,gs))

#for (n_order,g) in zip(ns,gs)
#let
for g in gs
  L= (4,4)
  N = prod(L);
  J = -1;
  h = -0.

  n_order = 3

  #array_id = parse(Int, ARGS[1])
  # (n_order,g) = p #params[array_id]

  ts = [0.0,0.1]
  step = 1.1
  tmax = 1e20

  ### logarithmic timesteps ###
  while true
    push!(ts, ts[end]*step)
    ts[end] > tmax && break
  end

  next_neighbours = nearest_neighbours(L, collect(1:prod(L)))

  spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
  dw_precalc = map(spin_basis) do spin
      return domainWallL(spin, L)
  end

  ### sort basis according to domain wall length ###
  sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
  dw_precalc  = [d[1] for d in sorted_spin_basis]
  spin_basis  = [d[2] for d in sorted_spin_basis]
  imb_precalc = map(s -> imbalance(s, L), spin_basis)

  spin_basis  = spin_basis[99:522]
  dw_precalc  = dw_precalc[99:522]
  imb_precalc = imb_precalc[99:522]

  ### initial state ###
  init_idx = 476 .- 98

  psi=zeros(length(spin_basis))
  psi[init_idx]=1
  psi = sparse(psi)

  vals,vecs,_ = readSpec("../data/spec_SWred_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

  ### map initial state onto eigenbasis of H ###
  psi = Transpose(vecs) * psi

  data = Any[]

  for (t, tf) in zip(ts[1:end-1], ts[2:end])
    dt = tf-t

    @time "time t= $(round(t, digits=1))" begin
      psi_prime = vecs*psi

      imb = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        α = abs2(psi_i)
        return α * imb_precalc[i]
      end

      # N      = real.(dot(psi_prime, H0      * psi_prime))
      # N_corr = real.(dot(psi_prime, H0_corr * psi_prime))
      #
      # D      = dot(psi_prime, D0*psi_prime)
      # D_corr = 0 #dot(psi, D0_corr*psi)
      # append!(data, [[t, imb, N, N_corr, D, D_corr]])

      append!(data, [[t, imb]])

      # Propagate state
      U = exp.(-1im*dt .* vals)
      psi = U .* psi
    end

    #vecs = sort([(i,p) for (i,p) in enumerate(abs2.(psi_prime)) if p>1e-3], by=x->x[2], rev=true)
    #@show [v[1] for v in vecs]
  end

  df = DataFrame(t = [real(d[1]) for d in data], imb = [real(d[2]) for d in data]) #, N = [real(d[3]) for d in data], N_corr = [real(d[4]) for d in data], D = [real(d[5]) for d in data], D_corr = [real(d[6]) for d in data])
  CSV.write("../data/obs_SWred_n=$(n_order)_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)

  #dfwf = DataFrame(psi = psi)
  #CSV.write("../data/wf_ED_longtime_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", dfwf);
end
