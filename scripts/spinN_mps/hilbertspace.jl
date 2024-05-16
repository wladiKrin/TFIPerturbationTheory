function ITensors.space(::SiteType"SpinN";
  nz = 2,
  conserve_qns = false,
  conserve_number = conserve_qns,
  qnname_number = "Number")

  #global GLB_N_BOSONS = nbosons
  if conserve_number
    return [QN(qnname_number, nn) => 1 for nn in -nz:nz]
  end 

  return 2*nz + 1 
end

function ITensors.state(::StateName{N}, ::SiteType"SpinN", s::Index) where{N}
  n = parse(Int, String(N))
  nd = dim(s)
  nb = div(nd - 1, 2)
  st = zeros(nd)
  st[n + nb + 1] = 1.0
  return itensor(st, s)
end

function ITensors.op(::OpName"Id", ::SiteType"SpinN", ds::Int...)
  d = prod(ds)
  # return diagm(fill(1.0, d))
  return Matrix(I, d, d)
end

ITensors.op(on::OpName"I", st::SiteType"SpinN", ds::Int...) = op(alias(on), st, ds...)
ITensors.op(on::OpName"F", st::SiteType"SpinN", ds::Int...) = op(OpName"Id"(), st, ds...)

function ITensors.op(::OpName"X", ::SiteType"SpinN", d::Int)    
  mat = zeros(d,d)

  for kk in 1:(d-1)
    mat[kk, kk+1] = 1
    mat[kk+1, kk] = 1
  end
  return mat
end

function ITensors.op(::OpName"S-", ::SiteType"SpinN", d::Int)    
  mat = zeros(d,d)

  for kk in 1:(d-1)
    mat[kk, kk+1] = 1
  end
  return mat
end

function ITensors.op(::OpName"S+", ::SiteType"SpinN", d::Int)    
  mat = zeros(d,d)

  for kk in 1:(d-1)
    mat[kk+1, kk] = 1
  end
  return mat
end

function ITensors.op(::OpName"N", ::SiteType"SpinN", d::Int)
  mat = zeros(d,d)
  nb = div(d-1, 2)
  for kk in 1:(d)
    mat[kk,kk] = (kk - nb - 1)
  end
  return mat
end

function ITensors.op(::OpName"Nabs", ::SiteType"SpinN", d::Int)
  mat = zeros(d,d)
  nb = div(d-1, 2)
  for kk in 1:(d)
    mat[kk,kk] = abs(kk - nb - 1)
  end
  return mat
end

function ITensors.op(::OpName"Imb", ::SiteType"SpinN", d::Int)
  mat = zeros(d,d)
  nb = div(d-1, 2)
  for kk in 1:(d)
    mat[kk,kk] = 1-abs(kk - nb - 1)/nb
  end
  return mat
end

function ITensors.op(::OpName"N2", ::SiteType"SpinN", d::Int)
  mat = zeros(d,d)
  nb = div(d-1, 2)
  for kk in 1:(d)
    mat[kk,kk] = (kk - nb - 1)^2
  end
  return mat
end

function ITensors.op(::OpName"Proj", ::SiteType"SpinN", d::Int; n::Int)
  mat = zeros(d,d)
  nb = div(d-1,2)
  abs(n) > nb && return mat
  pos = nb + n + 1
  mat[pos, pos] = 1
  return mat
end

function ITensors.op(on::OpName, st::SiteType"SpinN", s1::Index, s_tail::Index...; kwargs...)
  rs = reverse((s1, s_tail...))
  ds = dim.(rs)
  opname = string(ITensors.name(on))
  if occursin("Proj", opname)
    pos = parse(Int64, opname[5:end])
    opmat = op(OpName"Proj"(), st, ds...; n = pos, kwargs...)
  else
    opmat = op(on, st, ds...; kwargs...)
  end
  return itensor(opmat, prime.(rs)..., dag.(rs)...)
end
