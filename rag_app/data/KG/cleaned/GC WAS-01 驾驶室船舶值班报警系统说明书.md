[H1] 一、总述

驾驶台驾驶室船舶值班报警系统(Bridge Navigational Watch Alarm System)英文缩写为BNWAS，主要功能是防止驾驶员在航行值班时不能履行值班职责而使船舶处于无人操纵的危险局面发生。当上述情况出现时。系统将产生一系列逐步延伸的报警直到引起相关人员的注意，确保值班驾驶员履行其值班职责。

[H1] 一. Generalization

The main function of Bridge Navigational Watch Alarm System whose short name is BNWAS is toprevent ship from unmanned situation caused by the watcher is out of duty when watching. Whenthe above situation happens, the system will give series of stepwise extension alarm until draw therelevant person's attention and ensure the watcher is on duty.

本产品满足以下规范。
[H2] 1、IMO MSC 128(75)决议：驾驶室驾驶室船舶值班报警系统--BNWAS
[H2] 2、中国船级社电气电子产品型式认可试验指南(2006)
[H2] 3、CCS通函关于执行MSC.282(86)决议有关SOLAS II-1章修订内容(2011年1月1日生效)的通知

This product meets the following specifications:
[H2] 1、IMO MSC 128(75) Resolution: BNWAS
[H2] 2、CCS Electric Product Type Approval Test Guideline (2006)
[H2] 3、Notice on implement revising content for SOLAS II-1 Chapter of MSC.282(86) Resolution, which will be in force on January 1st, 2011

[H1] 4、型号命名方式
[H2] 4、Naming method of model.

河南光彩驾驶台航行值班报警系统/模块命名规则

[IMG path=data/KG/images/GC WAS-01 驾驶室船舶值班报警系统说明书/a17c303b1cfc6e6a22ac39c10c23430c4c95da904c9e59235690e42682e64d77.jpg alt=images/dd1861fe6c01459236076ac959d52de76f1b4313afda65a3b0fea74e7ab344bd.jpg]

1 厂商：河南光彩电器有限公司
2 ——产品类型：驾驶台航行值班报警系统
3 设计序列号为1
4 产品名称：

01 主控单元
02 输出接线单元
03 电源单元
04 报警延伸选择单元
05 驾驶室复位单元
06值班官员延伸单元
07 左右翼复位单元（防水）
08 热释传感器自动复位单元

5 ——子模块：

与功能代号4 配合使用表示该单元的子模块

举例：GC WAS-01-03/01

表示值班报警电源单元的电源接线模块

[H1] 二、技术参数 Technical Parameters:
[H2] 1、工作电压：DC24V（±20%-30%）2A；

Working Voltage:DC24V（±20%-30%）2A;
[H2] 2、工作环境： -10^{\circ}\mathrm{C} \sim +55^{\circ}\mathrm{C} ；

Working Atmosphere: -10^{\circ} \mathrm{C} \sim +55^{\circ} \mathrm{C} ;
[H2] 3、相对湿度： \leqslant \mathrm{RH}95\% (+40^{\circ}\mathrm{C}) ；

Relative Humidity: \leqslant RH95% (+40°C);
[H2] 4、电磁兼容性能：设备中有严密的抗干扰措施，满足相关规范要求；

Electromagnetic Compatibility: with strict ant jamming measurement, meet the

requirements of relevant specifications.
[H2] 5、设备具有防振及防潮湿、防盐雾、防霉菌的措施；

With measurements of anti-vibration, moisture, salt mist and mildew.
[H2] 6、自动舵遥控启动输入（无源触点）

Autopilot remote starting input(passive contact) 7、导航自动复位信号输入通道数 3（无源触点）

The channel number of navigation automatic reset signal input is 3 (passive contact)
[H2] 8、驾驶室复位输入输出通道数 3

The channel number of WH reset input/output is 3.
[H2] 9、延伸输出到官员舱室报警通道数 4

The channel number of alarm on extending output to officer cabin is 4.
[H2] 10、延伸输出到公共区域报警通道数 1

The channel number of alarm on extending output to public area is 1.
[H2] 11、延伸到通用报警输出：无源触点；

Extending to general alarm output: passive contact
[H2] 12、无源触点最大容量：DC36V/2A；

Maximum capacity of passive contact: DC36V/2A;
[H2] 13、VDR 输出：RS485

VDR output RS485
[H2] 14、防护等级：IP22；左右翼 IP56

Protection rate: IP22, two wings: IP56
[H2] 15、定时精确度
[H2] 1、$AYXDR, A 为头文件 stands for head document
[H2] 2、XX 报警名称 XX is alarm name
[H2] 3、c 为报警状态指示 “1” 为报警状态, ”0” 为非报警状态

c stands for alarm condition indication “1” for alarm condition, “0” for non-alarm condition.
[H2] 4、hh 为异或校验和 ASC 码

hh stands for XOR check and ASC code.

实际接受到的数据：actual receiving data:

AYXDR, A, BRIDGE WATCH ACTIVE, , 0*OD

AYXDR, A, BRIDGE WATCH STANDBY, , 1*07

AYXDR, A, BRIDGE WATCH MALFUNCTION, , 0*03

AYXDR, A, MANUAL_MODE, , 1*00

AYXDR, A, AUTO_MODE, , 0*04

AYXDR, A, VISUAL ALARM, , 1*01

AYXDR, A, AUDIO ALARM 1, , 1*02

AYXDR, A, AUDIO ALARM 2, , 1*01

AYXDR, A, AUDIO ALARM 3, , 1*00

AYXDR, A, EMERGENCY CALL, , 1*OD

[H1] 六、系统图 System drawing.

[IMG path=data/KG/images/GC WAS-01 驾驶室船舶值班报警系统说明书/04e4adb5a493daa85430153617c60d812930bf5956ba118da0bc7c97f7b64a83.jpg alt=images/fa290855d8b117a6e9c070e5f94cbeb3184ee21a5c8a94108374f6f93c71eebf.jpg]