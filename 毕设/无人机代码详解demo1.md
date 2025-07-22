这段代码实现了一个针对4架同质无人机在带通信约束和静止障碍物环境中搜索目标的仿真示例。下面对代码各部分进行详细分析：

---

## 1. 初始化部分

### a. 清空环境与变量
```matlab
clear;
close all;
clc ;
```
- **作用**：清除工作区变量、关闭所有图形窗口以及清空命令行，确保仿真从干净的状态开始。

### b. 地图信息定义
```matlab
map.x_min = 0;
map.x_max = 1600;
map.y_min = 0;
map.y_max = 800;
map.grid_size = 4;
```
- **说明**：定义了仿真地图的边界（x 和 y 方向的最小值和最大值）以及网格的大小，这对后续的搜索和路径规划非常关键。

### c. 仿真时间设置
```matlab
time.delta = 1;         % 仿真步长（单位：秒）
time.step = 0;          % 当前步数（初始为0）
time.step_max = 300;    % 最大仿真步数
time.t = 0;             % 当前仿真时间
```
- **说明**：设置了仿真的时间步长、当前步数、最大步数以及实际时间，用于控制整个仿真进程。

### d. 仿真显示参数
```matlab
property.sim.flag_plot = 1;         % 是否实时绘图
property.sim.plot_interval = 10;    % 绘图间隔（每10步绘制一次）
```
- **作用**：通过设置是否绘图及绘图间隔，用户可以观察到仿真过程中无人机的运动轨迹。

---

## 2. 无人机、目标、障碍物及遗传算法参数

### a. 无人机参数设置
```matlab
property.uav.num = 4;                     % 无人机数量
property.uav.initflag = 0;                % 初始化方式（0：手动，1：自动）
property.uav.vel_lim = [20,40];           % 速度限制
property.uav.acc_lim = [-2,2];            % 加速度限制
property.uav.yaw_lim = [-pi,pi];          % 航向角限制
property.uav.yaw_rate_lim = [-pi/7,pi/7];   % 航向角速率限制
property.uav.PID_roll = [15,0,0];         % Roll通道PID参数
property.uav.PID_pitch = [10,0,0];        % Pitch通道PID参数
property.uav.PID_yaw = [5,0,10];          % Yaw通道PID参数
property.uav.footprint_length = 100;      % 路径记录长度
property.uav.com_distance = 160;          % 通信范围
property.uav.com_type = 0;                % 通信模式（0：局部，1：全局）
property.uav.expert_system = 1;           % 是否采用专家系统（1：是）
property.uav.foresee_traj = 0;            % 是否预测其他无人机轨迹
property.uav.foresee_num = 3;             % 预测步数数量
property.uav.APF_distance = 200;          % 人工势场有效距离
property.uav.APF_param1 = 0.6*10e-2;      % 人工势场参数1（μ）
property.uav.APF_param2 = 10;             % 人工势场参数2（k）
property.uav.d_tar = 160;                 % 目标警戒距离
property.uav.d_obs = 30;                  % 障碍物警戒距离
property.uav.safe_xg = 2;                 % 距地图x边界的安全网格数
property.uav.safe_yg = 2;                 % 距地图y边界的安全网格数
property.uav.safe_obs = 20;               % 与障碍物的安全距离
property.uav.safe_uav = 30;               % 无人机之间的安全距离
property.uav.seeker_type = 1;             % 传感器视场类型
property.uav.seeker_pd = 0.80;            % 目标检测概率
property.uav.seeker_pf = 0.00;            % 目标误报概率
property.uav.search_interval = 1;         % 搜索算法调用间隔
property.uav.search_jump = 1;             % 跳格数（由专家系统最终决定）
property.uav.search_count = property.uav.search_interval/time.delta;
```
- **说明**：这一部分详细配置了无人机群的飞行特性、通信约束、目标检测、安全距离以及搜索策略参数。其中包括飞行控制（PID参数）、通信方式、以及采用跳格方法进行目标搜索等。

### b. 目标与障碍物参数
- **目标设置**：
  ```matlab
  property.tar.num = 5;                        % 目标数量
  property.tar.initflag = 0;                   % 初始化方式（手动/自动）
  property.tar.x_lim = [map.x_min,map.x_max];    % x方向位置限制
  property.tar.y_lim = [map.y_min,map.y_max];    % y方向位置限制
  property.tar.vel_lim = [20,40];               % 速度限制
  property.tar.acc_lim = [-2,2];                % 加速度限制
  property.tar.yaw_lim = [-pi,pi];              % 航向角限制
  property.tar.yaw_rate_lim = [-pi/7,pi/7];       % 航向角速率限制
  property.tar.footprint_length = 400;          % 路径记录长度
  ```
- **障碍物设置**：
  ```matlab
  property.obs.num = 6;                         % 障碍物数量
  property.obs.initflag = 0;                    % 初始化方式（手动/自动）
  property.obs.x_lim = [map.x_min,map.x_max];     % x方向位置限制
  property.obs.y_lim = [map.y_min,map.y_max];     % y方向位置限制
  property.obs.r_lim = [20,80];                  % 障碍物半径限制
  ```
- **说明**：通过这些参数设定了目标和障碍物在地图上的分布范围、运动属性和大小，保证仿真中目标和障碍物的合理生成和动态表现。

### c. 遗传算法参数
```matlab
property.GA.population_num = 100;   % 种群数量
property.GA.gene_length = 5;        % 基因长度（由专家系统最终决定）
property.GA.mp = 0.5;               % 变异概率
property.GA.cp = 0.5;               % 交叉概率
property.GA.iteration = 50;         % 最大迭代次数
```
- **说明**：遗传算法被用于优化搜索策略或路径规划，参数决定了种群规模、基因表示、以及算法中的交叉、变异策略和迭代次数。

---

## 3. 初始化对象

代码中使用了多个自定义函数来生成初始状态：
```matlab
UAV_Coordinate = Creat_UAV_Coordinate(property,map);
UAV_Control = Creat_UAV_Control(property.uav,UAV_Coordinate,map);
TAR = Creat_TAR(property.tar,map);
OBS = Creat_OBS(property.obs);
GS = Creat_Ground_Station(time,property,map,TAR);
UAV_Coordinate = Init_UAV_Map(UAV_Coordinate,GS);
```
- **说明**：
  - **Creat_UAV_Coordinate**：生成无人机的位置信息和状态结构数组，通常包括无人机当前坐标、速度、姿态等。
  - **Creat_UAV_Control**：初始化每架无人机的控制器参数，准备后续的飞行控制计算。
  - **Creat_TAR & Creat_OBS**：生成目标和障碍物的结构体数组。
  - **Creat_Ground_Station**：初始化地面站信息，用于管理和协调无人机数据。
  - **Init_UAV_Map**：初始化无人机的搜索信息，可能包括初始状态下对地图的覆盖情况等。

---

## 4. 主循环仿真

### a. 仿真步循环
```matlab
for step=1:time.step_max
    time.step = step;
    ...
end
```
- **说明**：整个仿真过程以步为单位进行，总步数为300步。每一步更新仿真状态和时间。

### b. 进度输出与绘图控制
```matlab
if print_count==10
    fprintf('Step number = %d, finised %d percent\n', step, step/time.step_max*100);
    print_count=0;
end

if property.sim.flag_plot
    plot_count=plot_count+1;
    if plot_count==property.sim.plot_interval
        if(step>property.sim.plot_interval)
            close(F);
        end
        F = Plot_UAV_Trajectory(map,GS,UAV_Coordinate,TAR,OBS);
        drawnow;
        plot_count=0;
    end
end
```
- **作用**：
  - 每10步输出一次进度提示，帮助用户了解仿真进度。
  - 根据设定的绘图间隔定时更新图形窗口，显示当前无人机轨迹、目标与障碍物位置。

### c. 分布式计算与无人机各自决策
```matlab
for i=1:property.uav.num
    UAV_Coordinate(i)= Trans_Mesg(UAV_Coordinate(i),UAV_Control(i),0);
    UAV_Coordinate(i)= UAV_Companion_Computer(time,i,UAV_Coordinate,TAR,OBS);
    UAV_Control(i)= Trans_Mesg(UAV_Coordinate(i),UAV_Control(i),1);
    UAV_Control(i)= UAV_Flight_Controller(UAV_Control(i));
end
```
- **说明**：
  - **信息传递（Trans_Mesg）**：该函数在伴随计算机（companion computer）和飞控控制器之间传递数据，模拟无人机内部的信息交互。
  - **伴随计算机计算（UAV_Companion_Computer）**：无人机在伴随计算机上进行目标搜索、路径规划或决策计算，考虑目标、障碍物和其他无人机信息。
  - **飞行控制器（UAV_Flight_Controller）**：根据伴随计算机的输出，更新无人机控制指令，调整姿态、速度等参数。

### d. 障碍物运动与地面站更新
```matlab
OBS = OBS_Movement(OBS,time,map);
GS = Com_Station(time,UAV_Coordinate,GS);
```
- **说明**：
  - **OBS_Movement**：更新障碍物位置（在本例中为静止障碍物，可能在其他demo中启用动态障碍物）。
  - **Com_Station**：更新地面站信息，将无人机数据传递给地面站，便于整体调度或数据记录。

---

## 5. 仿真结果的后处理与绘图

```matlab
Plot_UAV_Trajectory(map,GS,UAV_Coordinate,TAR,OBS);
Global_Proba = Plot_Data_Proba(time,map,property,UAV_Coordinate,TAR);
Global_Uncer = Plot_Data_Uncer(time,map,property,UAV_Coordinate,TAR);
```
- **作用**：
  - **Plot_UAV_Trajectory**：绘制最终无人机的飞行轨迹，展示搜索过程。
  - **Plot_Data_Proba** 与 **Plot_Data_Uncer**：分别展示全局目标检测概率和不确定性的数据变化，帮助分析搜索效率和可靠性。

---

## 总结

- **模块化设计**：代码通过将地图、时间、无人机、目标、障碍物、以及遗传算法等参数分模块定义，使得系统结构清晰，便于维护和扩展。
- **分布式仿真**：每架无人机独立进行信息处理和决策，通过伴随计算机与飞行控制器的交互，模拟了真实系统中信息传递和控制的过程。
- **实时绘图与数据输出**：通过定时输出仿真进度和实时绘图，用户能够直观地看到无人机搜索轨迹以及目标检测数据。
- **跳格搜索方法**：参数中包含跳格数设置，结合专家系统和遗传算法，表明该方法旨在提高搜索效率和全局优化能力。

这段代码不仅展示了无人机群协同搜索的基本框架，还体现了在通信约束、目标检测和避障等多重条件下如何进行分布式决策与控制。整个流程从初始化、状态更新、决策计算到结果绘制，都做了较为详细的设计，为进一步研究和扩展提供了良好的基础。

如果有其他细节问题或者需要进一步优化代码，可以继续讨论。