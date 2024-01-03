include("../src/SW.jl")

ns = [1,2,3]
gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-2.0]
#params = collect(Iterators.product(ns,gs))

# for (n_order,g) in zip(ns,gs)
let
  L= (4,4)
  N = prod(L);
  J = -1;
  h = -0.
  n_order = 3

  array_id = parse(Int, ARGS[1])
  g = gs[array_id]

  #(n_order,g) = params[array_id]

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
  init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
  init_idx = first(spin_basis_table[Tuple(init_spin)]);

  psi=zeros(length(spin_basis))
  psi[init_idx]=1
  psi = sparse(psi)

  ### build Hamiltonians ###
  H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
  H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
  V1 = H1+R1;

  #==================== perturbative =========================#
  vecH, vecR, vecS = SW_transformation(H0, H1, R1, domainWall_table, n_order)
  H_pert = dropzeros(reduce(+, vecH; init=H0))
  H_pert = H_pert[99:522,99:522]

  dw_precalc = dw_precalc[99:522]

  println("Starting caluclation of eigenvectors/-values")
  @time "eigen: " vals, vecs = eigen(Matrix(H_pert))
  dw = map(1:size(vecs)[2]) do i
      absVals = abs2.(vecs[:,i])
      return sum(absVals .* dw_precalc)
  end

  saveSpec("../data/spec_SWred_n=$(n_order)_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)", vals, vecs, dw);
end
