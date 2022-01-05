; n = 4
; # terms = 151
((lambda (f1) (let* ((d1 (f1 #t))) (f1 #f))) (lambda (x1) ((lambda (f2) (let* ((d2 (f2 #t))) (f2 #f))) (lambda (x2) ((lambda (f3) (let* ((d3 (f3 #t))) (f3 #f))) (lambda (x3) ((lambda (f4) (let* ((d4 (f4 #t))) (f4 #f))) (lambda (x4) ((lambda (z) (z x1 x2 x3 x4)) (lambda (y1 y2 y3 y4) y1))))))))))
