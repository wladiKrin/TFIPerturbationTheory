using Pkg; Pkg.activate(".")
using Revise
using Statistics
using CSV, DataFrames, HDF5
using LinearAlgebra

using ITensors
using ITensorTDVP
using Observers

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

function measure_N(; state)
  res = real.(collect(expect(state, "N")))
  # println("N: $(res)")
  return res
end

function measure_proj0(; state)
    return real.(collect(expect(state, "Proj0")))
end
function measure_proj1(; state)
    return real.(collect(expect(state, "Proj1"))) .+ real.(collect(expect(state, "Proj-1")))
end
function measure_proj2(; state)
    return real.(collect(expect(state, "Proj2"))) .+ real.(collect(expect(state, "Proj-2")))
end
function measure_proj3(; state)
    return real.(collect(expect(state, "Proj3"))) .+ real.(collect(expect(state, "Proj-3")))
end
function measure_proj4(; state)
    return real.(collect(expect(state, "Proj4"))) .+ real.(collect(expect(state, "Proj-4")))
end
function measure_proj5(; state)
    return real.(collect(expect(state, "Proj5"))) .+ real.(collect(expect(state, "Proj-5")))
end
function measure_proj6(; state)
    return real.(collect(expect(state, "Proj6"))) .+ real.(collect(expect(state, "Proj-6")))
end
function measure_proj7(; state)
    return real.(collect(expect(state, "Proj7"))) .+ real.(collect(expect(state, "Proj-7")))
end
function measure_proj8(; state)
    return real.(collect(expect(state, "Proj8"))) .+ real.(collect(expect(state, "Proj-8")))
end

obs = observer(
    "N" => measure_N, 
    "Nabs" => measure_proj0, 
    "N2" => measure_proj1, 
)

@show obs, typeof(obs)

#   N = 10 
#   D = 1 
#   maxDim = 64 
#   (J,h) = (-1.,-0.)
#   g = -2.0 
#
#   tmax = 100
#   dt = 0.1
#
#   s = siteinds("SpinN", N; nz = D) #, conserve_qns=true)
#
#   # Model MPO
#   # model = Ising(L=N, g=g, J = J, d=D)
#   # H = MPO(model, s)
#
#   psiRand = randomMPS(s, "0"; linkdims=64)
#   psi0    = MPS(s, "0")
#
#
#   println("fock")
#   @show measure_proj0(state=psi0)
#   for (i,v) in enumerate(psi0)
#       @show i, size(array(v))
#   end
#   @show psi0[2]
#
#   println("rand")
#   @show measure_proj0(state=psiRand)
#   psiRand = randomMPS(s; linkdims=64)
#   for (i,v) in enumerate(psiRand)
#       if i == 1
#           mat = fill(0, size(array(psiRand[i])))
#           mat[1,D+1] = 1
#           psiRand[i] = ITensor(mat, inds(psiRand[i]))
#       elseif i == N
#           mat = fill(0, size(array(psiRand[i])))
#           mat[D+1,1] = 1
#           psiRand[i] = ITensor(mat, inds(psiRand[i]))
#       else i == N
#           mat = fill(0, size(array(psiRand[i])))
#           mat[D+1,1,1] = 1
#           psiRand[i] = ITensor(mat, inds(psiRand[i]))
#       end
#   end
