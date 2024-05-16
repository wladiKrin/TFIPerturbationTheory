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

# util function to save data in obs
function savedata(name::String, obs)
    name == "" && return
    @show obs
    h5open(name*".h5", "w") do file    
        # iterate through the fields of obs and append the data to the dataframe
        names = ["sweep","N","Nabs","N2","proj0","proj1","proj2","proj3","proj4"]
        for n in names
            create_group(file, n)
        
            for (i,data) in enumerate(obs[:,n])
                file[n][string(i)] = data
            end
        end 
        file["energy"]  = obs[:,"energy"]
        file["ent"] = obs[:,"ent"]
    end
end

function Ising(;L, g, J = -1, d)
  ampo = OpSum()
  for (j1,j2) in zip(1:L, vcat(2:L, 1))
    # ampo += (g,"XMin", j1) 
    ampo += (g,"S+", j1) 
    ampo += (g,"S-", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end

  return ampo
end

function current_sweep(; which_sweep)
  return which_sweep
end

function measure_N(; state)
  return real.(collect(expect("N", state)))
end
function measure_entropy(; state)
  pos = floor(Int,length(vertices(state))/2)
  psi_ = orthogonalize(state, pos)
  _,S,_ = svd(psi_[pos], (commonind(psi_[pos-1], psi_[pos]), filter(i->hastags(i, "Site"), inds(psi_[pos]))))
  return mapreduce(+, 1:dim(S,1)) do i
    p = S[i,i]^2
    return -p*log(p)
  end
end
function measure_Nabs(; state)
  return real.(collect(expect("Nabs", state)))
end
function measure_N2(; state)
  return real.(collect(expect("N2", state)))
end
function measure_proj0(; state)
  return real.(collect(expect("Proj0", state)))
end
function measure_proj_1(; state)
  res = real.(collect(expect("Proj-1", state)))
  println("-1: $(mean(res))")
  return res
end
function measure_proj_2(; state)
  return real.(collect(expect("Proj-2", state)))
end
function measure_proj_3(; state)
  return real.(collect(expect("Proj-3", state)))
end
function measure_proj_4(; state)
  return real.(collect(expect("Proj-4", state)))
end
function measure_proj1(; state)
  res = real.(collect(expect("Proj1", state)))
  println("1: $(mean(res))")
  return res
end
function measure_proj2(; state)
  return real.(collect(expect("Proj2", state)))
end
function measure_proj3(; state)
  return real.(collect(expect("Proj3", state)))
end
function measure_proj4(; state)
  return real.(collect(expect("Proj4", state)))
end
function measure_proj5(; state)
  return real.(expect("Proj5", state) .+ expect("Proj-5", state))
end
function measure_proj6(; state)
  return real.(expect("Proj6", state) .+ expect("Proj-6", state))
end
function measure_proj7(; state)
  return real.(expect("Proj7", state) .+ expect("Proj-7", state))
end
function measure_proj8(; state)
  return real.(expect("Proj8", state) .+ expect("Proj-8", state))
end

function measure_projSum(; state)
  res = real.(collect(expect("Proj-4", state) .+ expect("Proj-3", state) .+ expect("Proj-2", state) .+ expect("Proj-1", state) .+ expect("Proj0", state) .+ expect("Proj1", state) .+ expect("Proj2", state) .+ expect("Proj3", state) .+ expect("Proj4", state)))
  println("sum: $(mean(res))")
  return mean(res)

end

# Ls = (8,16,20,24)
# bondDims = (16,32)
# gs = [-0.25,-0.5,-0.6,-0.75,-0.8,-0.85,-0.9,-1.0,-1.15,-1.25,-1.5]
gs = [-0.25,-0.5,-0.75,-0.9,-1.,-1.25,-1.5]
gs = [-0.9]

let
  N = parse(Int, ARGS[1]) #length of lattice
  D = 4 #max Sz component
  maxDim = parse(Int, ARGS[2])
  (J,h) = (-1.,-0.)
  g = gs[parse(Int, ARGS[3])]

  tmax = 50
  dt = 0.1

  ## Ising model ##
  graph = named_path_graph(N)
  # in = IndsNetwork(graph)

  s = siteinds("SpinN", graph; nz = D, conserve_qns=true)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)
  H = TTN(model, s)
  # H = ITensors.MPO(model, s)

  # Make MPS
  # states = ["0","0","1","2","2","1","0","0"]
  states = ["0","0","0","0","0","0","0","0"]
  psi = TTN(ITensorNetwork(s, x->states[x]))

  function measure_En(; state)
    return real(inner(state', H, state))
  end
  
  obs = observer(
    "sweep" => current_sweep, 
    "energy" => measure_En, 
    "N" => measure_N, 
    "ent" => measure_entropy, 
    "Nabs" => measure_Nabs, 
    "N2" => measure_N2, 
    "proj-4" => measure_proj_4,
    "proj-3" => measure_proj_3,
    "proj-2" => measure_proj_2,
    "proj-1" => measure_proj_1,
    "proj0" => measure_proj0,
    "proj1" => measure_proj1,
    "proj2" => measure_proj2,
    "proj3" => measure_proj3,
    "proj4" => measure_proj4,
    "projSum" => measure_projSum,
    # "proj5" => measure_proj5,
    # "proj6" => measure_proj6,
    # "proj7" => measure_proj7,
    # "proj8" => measure_proj8,
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

  # df = DataFrame(
  #   t     = obs.sweep .* dt, 
  #   en    = obs.energy, 
  #   N     = obs.N, 
  #   Nabs  = obs.Nabs, 
  #   N2    = obs.N2, 
  #   proj0 = obs.proj0, 
  #   proj1 = obs.proj1, 
  #   proj2 = obs.proj2, 
  #   proj3 = obs.proj3, 
  #   proj4 = obs.proj4, 
  #   # proj5 = obs.proj5, 
  #   # proj6 = obs.proj6, 
  #   # proj7 = obs.proj7, 
  #   # proj8 = obs.proj8, 
  # )

  # CSV.write("../../data/obs_hardcoreBosons_mps_test_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).csv", df)
  savedata("../../data/obs_hardcoreBosons_mps_test_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax)", obs)
  psimps = MPS(collect(vertex_data(psi)))
  h5open("./ttns/psi_Buldge_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).h5", "w") do file
    write(file, "mps", psimps)
  end
end
