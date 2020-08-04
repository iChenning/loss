import torch
import torch.nn as nn

img = torch.tensor([[[[1.0, 2, 3, 4, 5, 6, 7],
                      [2, 3, 4, 5, 6, 7, 8],
                      [3, 4, 5, 6, 7, 8, 9],
                      [2, 3, 4, 5, 6, 7, 8],
                      [1, 2, 3, 4, 5, 6, 7]]],
                    [[[1.0, 2, 3, 4, 5, 6, 7],
                      [2, 3, 4, 5, 6, 7, 8],
                      [3, 4, 5, 6, 7, 8, 9],
                      [2, 3, 4, 5, 6, 7, 8],
                      [1, 2, 3, 4, 5, 6, 7]]]
                    ])

m_all = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), bias=False)
ZeroPad = nn.ZeroPad2d(padding=(1,1,1,1))
img_new = ZeroPad(img)
out_all = m_all(img_new)

m_21 = nn.Conv2d(1, 1, kernel_size=(2, 3), stride=1, bias=False)
ZeroPad = nn.ZeroPad2d(padding=(1,1,1,0))
img_new = ZeroPad(img)
out_21 = m_21(img_new)

m_22 = nn.Conv2d(1, 1, kernel_size=(3, 2), stride=1, bias=False)
ZeroPad = nn.ZeroPad2d(padding=(0,1,1,1))
img_new = ZeroPad(img)
out_22 = m_22(img_new)

m_23 = nn.Conv2d(1, 1, kernel_size=(2, 3), stride=1, bias=False)
ZeroPad = nn.ZeroPad2d(padding=(1,1,0,1))
img_new = ZeroPad(img)
out_23 = m_23(img_new)

m_24 = nn.Conv2d(1, 1, kernel_size=(3, 2), stride=1, bias=False)
ZeroPad = nn.ZeroPad2d(padding=(1,0,1,1))
img_new = ZeroPad(img)
out_24 = m_24(img_new)

out_merge = torch.cat([out_all, out_21, out_22,out_23,out_24],dim=1)

m = nn.Conv2d(5,1,kernel_size=(3,3),stride=1,padding=1, bias=False)
out = m(out_merge)
print(out.shape)
print(out)