# Transportation-Network-Analysis-Course-Code
研究生课程《运输网络分析》课件上一些算法的python代码。

## 简介
研一上学期选修了一门《运输网络分析》课程，课件中涉及到和交通运输网络相关的图论算法，包括**最短路径、最小代价流以及交通流分配**（SO和UE）算法。其中一些算法在python的[networkx](https://networkx.github.io/documentation/stable/tutorial.html)库中已有定义，完全可以直接调用。**仅出于个人学习目的**，使用`python`编程实现其中一部分算例，尽管课程本身对编程未作任何要求。

## Todo
- 学习doctest，实现算例自动测试。
- 完善文档说明。
- 最短路径算法其他算法优化。
- 几个没搞定的算法：
  - 最小代价流的“消圈法”，在查找负权环时算法写的不对
  - Network Simplex算法
