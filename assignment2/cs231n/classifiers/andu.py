N,C,H,W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
   F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
   stride, pad = conv_param['stride'], conv_param['pad']
   H1 = int(1 + (H + 2 * pad - HH) / stride)
   W1 = int(1 + (W + 2 * pad - WW) / stride)
   out = np.zeros((N,F,H1,W1))
   # zero padding
   x_padding = np.zeros((N,C,H+pad*2,W+pad*2))
   for n in range(N):
       for c in range(C):
           x_padding[n,c] = np.pad(x[n,c],pad,'constant',constant_values=(0))
   # forward propagation
   for n in range(N):
       for f in range(F):
           for i in range(H1):
               for j in range(W1):
                   tmp_x = x_padding[n,:,stride*i:stride*i+HH, stride*j:stride*j+WW]
                   curr = tmp_x*w[f]
                   out[n,f,i,j] = np.sum(curr)+b[f]











                       x, w, b, conv_param = cache
                       N,C,H,W = x.shape
                       F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
                       stride, pad = conv_param['stride'], conv_param['pad']
                       H1 = int(1 + (H + 2 * pad - HH) / stride)
                       W1 = int(1 + (W + 2 * pad - WW) / stride)
                       dx = np.zeros(x.shape)
                       dw = np.zeros(w.shape)
                       db = np.zeros(b.shape)
                       # zero padding
                       x_padding = np.zeros((N,C,H+pad*2,W+pad*2))
                       dx_padding = np.zeros(x_padding.shape)
                       for n in range(N):
                           for c in range(C):
                               x_padding[n,c] = np.pad(x[n,c],pad,'constant',constant_values=(0))
                       # back propagation
                       for n in range(N):
                           for f in range(F):
                               for i in range(H1):
                                   for j in range(W1):
                                       dx_padding[n,:,stride*i:stride*i+HH, stride*j:stride*j+WW] += dout[n,f,i,j]*w[f]
                                       dw[f] += dout[n,f,i,j]*x_padding[n,:,stride*i:stride*i+HH, stride*j:stride*j+WW]
                                       db[f] += dout[n,f,i,j]

                       # remove padding
                       for n in range(N):
                           for c in range(C):
                               dx[n,c] = dx_padding[n,c,1:1+H,1:1+W]
