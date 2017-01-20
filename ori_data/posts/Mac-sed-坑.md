---
title: Mac sed 坑
date: 2016-01-16 10:47:46
tags:
---
## 坑1 
执行
```bash
sed -i "s/xxx/yyy/g" filename
```
<!-- more -->
出现错误
```
sed: 1:  invalid command code B
```
这种在Linux的用法不可以在Mac下直接使用，需要加上''
```bash
sed -i '' "s/xxx/yyy/g" filename (不需要备份)
sed -i '.bak' "s/xxx/yyy/g" filename (需要备份，备份为filename.bak)
```

## 坑2 
上述命令执行后仍有错误
```bash
sed: RE error: illegal byte sequence
```
需要在shell中加上：
```
export LC_COLLATE='C'
export LC_CTYPE='C'
```
---
博客地址：[52ml.me](http://www.52ml.me)<br>
原创文章，版权声明：自由转载-非商用-非衍生-保持署名 | [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)
<br>
---

