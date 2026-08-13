# Third-Party Notices

This project vendors selected algorithm cores. Upstream licenses, when available, and exact source commits are recorded in `configs/official_sources.yaml` and in source headers under `src/official/`.

## MIT-Licensed Components

| Component | Upstream commit | Copyright |
| --- | --- | --- |
| TENT | `e9e926a668d85244c66a6d5c006efbd2b82e83e8` | Copyright (c) 2021 Dequan Wang and Evan Shelhamer |
| EATA | `f739b3668cc7617e9b9f1979c1a358497a3472c3` | Copyright (c) 2023 Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan |
| CoTTA | `c212a204b32be4005092e4323105a24a29ad2952` | Copyright (c) 2021 Qin Wang |
| RoTTA | `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9` | Copyright (c) 2023 BITDA @ Beijing Institute of Technology |
| T2A | `33c8ccc64afdda260564123d6c790d030a89ff81` | Copyright (c) 2025 HongHanh2104 |

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## LAME

The vendored LAME core is adapted from commit `d2e5f63090bc1c8129bf7cbd781029a5955e1a67` of <https://github.com/fiveai/LAME> and remains under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Reuse must provide attribution, be non-commercial, and distribute adaptations under the same license. The complete license text is available at <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>.

## IAPL

The IAPL runtime model core is vendored from <https://github.com/liyih/IAPL> commit `a173e7783bbafaa00d60e6e31774a0bc14411a23` under `src/official/iapl/`. The upstream repository did not declare a software license when these files were imported. No open-source redistribution grant is claimed for this component; downstream users must obtain any required permission from the IAPL copyright holders.

## OST

The OST MetaXception, inner-loop optimizer, AM-Softmax, and extracted training and inference cores are derived from <https://github.com/liangchen527/OST> commit `1e4518b9e560baf9c5693f13a402fa5d7104190f` under `src/official/ost/`. The upstream OST repository did not declare a repository-wide software license when these files were imported. The AM-Softmax source itself carries an Apache-2.0 notice, but no broader license is inferred for the other OST files. No open-source redistribution grant is claimed for those files; downstream users must obtain any required permission from the OST copyright holders.
