#!/bin/bash
#
# b@20210314
#
unalias gs
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -dQUIET -sOutputFile=output.pdf input.pdf
#
#-dPDFSETTINGS params descri.参数描述：
#/prepress ：best, biggest 质量最好，文件最大
#/printer ：printer mode, recommended 打印模式，推荐压缩
#/ebook ： for ebook/text 电子书模式，适合文本
#/screen ：smallest, not recommended文件最小，不推荐
