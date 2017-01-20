---
title: some tips
date: 2016-01-22 15:57:24
tags: tips
---
## adb模拟点击
ADB中通过input来实现，用于输入touch，key等事件：
The sources are: trackball joystick touchnavigation mouse keyboard gamepad touchpad dpad stylus touchscreen
<!--more-->
The commands and default sources are:
      text <string> (Default: touchscreen)
      keyevent [--longpress] <key code number or name> ... (Default: keyboard)
      tap <x> <y> (Default: touchscreen)
      swipe <x1> <y1> <x2> <y2> [duration(ms)] (Default: touchscreen)
      press (Default: trackball)
      roll <dx> <dy> (Default: trackball)
```shell
adb shell input touchscreen  tap 10 10
```
## adb+python截图
screencap 本身支持标准输出，所以可以用管道符链接。但是 adb shell 会将结果中的 LF 转换为 CR+LF，会将 png 的格式破坏。于是这里将LF前的CR移除。
```shell
adb shell screencap -p | sed 's/\r$//' > screen.png
adb shell screencap -p | perl -pe 's/\x0D\x0A/\x0A/g' > screen.png   
```


```python
import subprocess
returnValue = subprocess.check_output(["adb", "shell", "screencap",  "-p"])
returnValue.replace('\x0D\x0A', '\x0A')
```

```python
#!/usr/bin/python
import subprocess
import time
for i in range(5):
    tic = time.time()
    subprocess.call(["adb", "shell", "input", "touchscreen", "tap", "10", "10"])
    toc = time.time()
    tic1 = time.time()
    returnValue = subprocess.check_output(["adb", "shell", "screencap",  "-p"]).replace('\x0D\x0A', '\x0A')
    toc1 = time.time()
    print 'time = ', toc-tic
    print 'time1 = ', toc1-tic1
    #returnValue = returnValue.replace('\x0D\x0A', '\x0A')
    fileName = 'screenshot-%d.png' % i
    screenshot = open(fileName, 'w')
    screenshot.write(returnValue)
    screenshot.close()

```
---
博客地址：[52ml.me](http://www.52ml.me)<br>
原创文章，版权声明：自由转载-非商用-非衍生-保持署名 | [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)
<br>
---

