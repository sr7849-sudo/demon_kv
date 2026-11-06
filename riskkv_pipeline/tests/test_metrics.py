from riskkv.metrics import exact_match, token_f1, rouge_l

def test_exact(): assert exact_match('The cat.','cat')==1.0

def test_f1_bounds():
    x=token_f1('alpha beta','alpha gamma'); assert 0 < x < 1

def test_rouge_bounds():
    x=rouge_l('a b c','a b d'); assert 0 <= x <= 1
