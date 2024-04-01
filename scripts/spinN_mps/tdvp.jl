using Pkg; Pkg.activate(".")
using Revise
using Statistics
using CSV, DataFrames, HDF5
using LinearAlgebra
using Observers

using ITensors
using ITensorNetworks
using NamedGraphs

include("hilbertspace.jl")

function Ising(;L, g, J = -1, d)
  ampo = OpSum()
  for (j1,j2) in zip(1:L, vcat(2:L, 1))
    ampo += (g,"X", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end

  return ampo
end

Ls = (8,16,20)
bondDims = (16,32)
gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

function current_sweep(; which_sweep)
  return which_sweep
end

function measure_N(; state)
  return real(mean(expect("N", state)))
end
function measure_Nabs(; state)
  return real(mean(expect("Nabs", state)))
end
function measure_N2(; state)
  return real(mean(expect("N2", state)))
end
function measure_proj0(; state)
  return real(mean(expect("Proj0", state)))
end
function measure_proj1(; state)
  return real(mean(expect("Proj1", state)) + mean(expect("Proj-1", state)))
end
function measure_proj2(; state)
  return real(mean(expect("Proj2", state)) + mean(expect("Proj-2", state)))
end
function measure_proj3(; state)
  return real(mean(expect("Proj3", state)) + mean(expect("Proj-3", state)))
end

# Ls = (8,16,20,24)
# bondDims = (16,32)
gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

let
  N = parse(Int, ARGS[1]) #length of lattice
  D = 4 #max Sz component
  maxDim = parse(Int, ARGS[2])
  (J,h) = (-1.,-0.)
  g = gs[parse(Int, ARGS[3])]

  tmax = 50.
  dt = 0.1

  ## Ising model ##
  graph = named_path_graph(N)
  # in = IndsNetwork(graph)

  s = siteinds("SpinN", graph; nz = D)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)
  H = TTN(model, s)
  # H = ITensors.MPO(model, s)

  # Make MPS
  psi = TTN(ITensorNetwork(s, "0"))

  function measure_En(; state)
    return real(inner(state', H, state))
  end
  
  obs = observer(
    "sweep" => current_sweep, 
    "energy" => measure_En, 
    "N" => measure_N, 
    "Nabs" => measure_Nabs, 
    "N2" => measure_N2, 
    "proj0" => measure_proj0,
    "proj1" => measure_proj1,
    "proj2" => measure_proj2,
    "proj3" => measure_proj3,
  )

  psi = tdvp(
    H,
    -im * tmax,
    psi;
    time_step = -im * dt,
    reverse_step=true,
    normalize=false,
    maxdim=maxDim,
    cutoff=1e-10,
    outputlevel=1,
    (sweep_observer!)=obs,
  )

  df = DataFrame(
    t     = obs.sweep .* dt, 
    en    = obs.energy, 
    N     = obs.N, 
    Nabs  = obs.Nabs, 
    N2    = obs.N2, 
    proj0 = obs.proj0, 
    proj1 = obs.proj1, 
    proj2 = obs.proj2, 
    proj3 = obs.proj3, 
  )

  CSV.write("../../data/obs_hardcoreBosons_mps_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).csv", df)
  psimps = MPS(collect(vertex_data(psi)))
  h5open("./ttns/psi_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).h5", "w") do file
    write(file, "mps", psimps)
  end
end
