using Pkg; Pkg.activate(".")
using Revise
using Statistics
using CSV, DataFrames, HDF5
using LinearAlgebra
using Observers

using ITensors
# using ITensorNetworks
using ITensorTDVP
using NamedGraphs

include("hilbertspace.jl")

function Ising(;L, g, J = -1, d)
  ampo = OpSum()
  # for (j1,j2) in zip(1:L, vcat(2:L, 1))
  for (j1,j2) in zip(1:L-1, 2:L)
    # ampo += (g,"X", j1) 
    ampo += (g,"S+", j1) 
    ampo += (g,"S-", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end
  ampo += (g,"S+", L) 
  ampo += (g,"S-", L) 

  return ampo
end

BLAS.set_num_threads(1)

Ls = (8,16,24,32)
bondDims = (16,32,64,128)
Szs = (4,8,12)

data = []

for (N, maxDim, D) in Iterators.product(Ls, bondDims, Szs)
  @show N, maxDim, D
  (J,h) = (-1.,-0.)
  g = -2.0 #gs[parse(Int, ARGS[3])]

  tmax = 0.5
  dt = 0.1

  ## Ising model ##
  graph = N #named_path_graph(N)

  s = siteinds("SpinN", graph; nz = D) #, conserve_qns=true)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)
  H = MPO(model, s)

  # Make MPS
  # states = ["0","0","1","2","2","1","0","0"]
  # states = ["0","0","0","0","0","0","0","0"]
  psi = randomMPS(s; linkdims=maxDim)

  function measure_En(; state)
    res = real(inner(state', H, state))
    println("E: $(res)")
    return res
  end
  
  t = 0 

  psi = tdvp(
    H,
    -im * dt,
    psi;
    nsweeps = 1,
    nsite = 1,
    reverse_step=true,
    normalize=false,
    maxdim=maxDim,
    cutoff=1e-10,
    outputlevel=1,
  )

  timing = @elapsed tdvp(
    H,
    -im * dt,
    psi;
    nsweeps = 5,
    nsite = 1,
    reverse_step=true,
    normalize=false,
    maxdim=maxDim,
    cutoff=1e-10,
    outputlevel=1,
  )

  push!(data, [N, maxDim, D, timing/5])
  df = DataFrame(
    L      = [d[1] for d in data],
    maxDim = [d[2] for d in data],
    Sz     = [d[3] for d in data],
    timing = [d[4] for d in data],
  )

  CSV.write("../../data/timings_mps_eff_model.csv", df)
end
