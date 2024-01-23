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


function Ising(J, g, h, N,s)
    ampo = OpSum()
	    
  for (j1,j2) in zip(1:N, vcat(2:N,1))
    # ampo += J, "Nabs", (j1, j2)
    add!(ampo, op("Nabs", s[j1], s[j2]))
	end

  for j in 1:N
    ampo += g, "Adag", j
    ampo += g, "A", j
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
  model = Ising(J, g, h, N,s)
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
