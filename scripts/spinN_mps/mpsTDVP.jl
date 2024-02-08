using Revise
using CSV, DataFrames

using ITensors
using ITensorTDVP

function ITensors.op(::OpName"Nabs", ::SiteType"Qudit", s1::Index, s2::Index)
  inds = vcat(collect(Iterators.product(1:dim(s1), 1:dim(s2)))...)
  mat = map(Iterators.product(inds,inds)) do ((iIn,jIn),(iOut,jOut))
    if iIn == iOut && jIn == jOut
      return (2*length(s1)-3) - 2 * abs(iIn-jIn)
    end
    return 0
  end
  return itensor(mat, s2', s1', s2, s1)
end

function ITensors.op(::OpName"AAdag", ::SiteType"Qudit", s::Index)
  mat = diagm(1 => fill(1,dim(s)-1), -1 => fill(1,dim(s)-1))
  return itensor(mat, s', s)
end


function Ising(J, g, h, N, locDim, s)
  ampo = OpSum()

  n = locDim-1

  function ITensors.op(::OpName"N2", ::SiteType"Qudit", s::Index)
    mat = diagm(collect(0:dim(s)-1).^2)
    return itensor(mat, s', s)
  end
  A = map(Iterators.product(1:n,1:n)) do (i,j)
    return i^(2*j)
  end
  b = collect(1:n)
  Ainv = A^-1
  alphas = Ainv*b
	    
  for (j1,j2) in zip(1:N, vcat(2:N,1))
    for (ia, a) in enumerate(alphas)
      ia = 2*ia
      for k in 0:ia
        opString = vcat(vcat(fill(["N", j1], k)...), vcat(fill(["N", j2], ia-k)...))
        coeff = J*a*binomial(ia, k)
        add!(ampo, coeff, opString...)
      end
	end

  for j in 1:N
    ampo += g, "AAdag", j
	end

	return ampo
end

let
  #parameters
  N = 8
  D = 8
  (J,g,h) = (-1.,-0.5,-0.)
  T = 10.
  dt = 0.1

  ## Ising model ##
  s = siteinds("Qudit", N; dim = D)
  
  # Model MPO
  model = Ising(J, g, h, N, D-1)
  # Make MPO
  mpo = ITensors.MPO(model, s)

  # states = vcat(fill("X+", Int(N/2)), fill("X-", Int(N/2)))
  psi = MPS(s, "4")

  phi = tdvp(
    H,
    -im * T,
    psi;
    time_step = -im * dt,
    nsweeps=2,
    reverse_step=true,
    normalize=false,
    maxdim=30,
    cutoff=1e-10,
    outputlevel=1,
  )
end
