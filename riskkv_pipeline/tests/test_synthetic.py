from riskkv.synthetic import make_synthetic

def test_shape_and_full():
 d=make_synthetic(20); assert len(d)==100; assert (d[d.policy=='full'].degradation==0).all()
