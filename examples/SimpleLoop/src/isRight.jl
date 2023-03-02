"""
Filter data for `x = r*s + b -y <= y` to get uniqe data points.
"""
function isRight(s,r,y; b)
  x = r*s + b -y
  return x > y
end
