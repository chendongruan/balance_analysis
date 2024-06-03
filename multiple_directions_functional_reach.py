import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import plotly.graph_objs as go
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# 设置中文字体
# 确保字体文件路径正确
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path)
rcParams['font.sans-serif'] = [font_prop.get_name()]
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Streamlit 应用程序标题
st.title('多方向平衡功能评估')

# 用户输入部分
st.sidebar.header('OpenCap信息输入')
uploaded_file = st.sidebar.file_uploader("trc文件位于MarkerData文件夹里。", type=["trc"])

# 展示和重置按钮并排放置
display_button = st.sidebar.button("展示")

# 处理上传的文件
if uploaded_file is not None and display_button:
    # 读取文件的所有行
    lines = uploaded_file.readlines()
    lines = [line.decode('utf-8') for line in lines]

    # 提取关节名称（忽略空白列）
    joint_names = [name for name in lines[3].strip().split('\t')[2:] if name]
    # 提取列名（忽略空白列）
    columns = [col for col in lines[4].strip().split('\t') if col]

    # 从第7行开始是数据
    data_lines = lines[6:]

    # 读取数据部分
    data = []
    for line in data_lines:
        if line.strip():
            data.append(line.strip().split('\t')[2:])  # 跳过前两个元素

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 转换数据类型为浮点数
    df = df.astype(float)

    # 创建一个字典来存储每个关节的数据
    joint_data = {}
    num_joints = len(joint_names)
    for i in range(num_joints):
        joint = joint_names[i]
        joint_columns = columns[i*3:i*3+3]
        joint_data[joint] = df[joint_columns]

    # 计算各个身体部分的质量
    Neck_mass = joint_data['Neck'].values * 0.0668
    Trunk_mass = ((joint_data['RShoulder'].values + joint_data['LShoulder'].values) / 2 + 
                  ((joint_data['RHip'].values + joint_data['LHip'].values) / 2 - 
                   (joint_data['RShoulder'].values + joint_data['LShoulder'].values) / 2) * 0.3782) * 0.4258
    RUpperarm_mass = (joint_data['RShoulder'].values + 
                      (joint_data['RElbow'].values - joint_data['RShoulder'].values) * 0.5754) * 0.0255
    LUpperarm_mass = (joint_data['LShoulder'].values + 
                      (joint_data['LElbow'].values - joint_data['LShoulder'].values) * 0.5754) * 0.0255
    RForearm_mass = (joint_data['RElbow'].values + 
                     (joint_data['RWrist'].values - joint_data['RElbow'].values) * 0.4559) * 0.0138
    LForearm_mass = (joint_data['LElbow'].values + 
                     (joint_data['LWrist'].values - joint_data['LElbow'].values) * 0.4559) * 0.0138
    RThigh_mass = (joint_data['RHip'].values + 
                   (joint_data['RKnee'].values - joint_data['RHip'].values) * 0.3612) * 0.1478
    LThigh_mass = (joint_data['LHip'].values + 
                   (joint_data['LKnee'].values - joint_data['LHip'].values) * 0.3612) * 0.1478
    RShank_mass = (joint_data['RKnee'].values + 
                   (joint_data['RAnkle'].values - joint_data['RKnee'].values) * 0.4352) * 0.0481
    LShank_mass = (joint_data['LKnee'].values + 
                   (joint_data['LAnkle'].values - joint_data['LKnee'].values) * 0.4352) * 0.0481
    RFoot_mass = (joint_data['RAnkle'].values + 
                  (joint_data['RBigToe'].values - joint_data['RAnkle'].values) * 0.4014) * 0.0129
    LFoot_mass = (joint_data['LAnkle'].values + 
                  (joint_data['LBigToe'].values - joint_data['LAnkle'].values) * 0.4014) * 0.0129
    Total_mass = (Neck_mass + Trunk_mass + RUpperarm_mass + LUpperarm_mass + 
                  RForearm_mass + LForearm_mass + RThigh_mass + LThigh_mass + 
                  RShank_mass + LShank_mass + RFoot_mass + LFoot_mass)

    # 计算左脚中点
    Mid_l_foot = ((joint_data['LBigToe'].values + joint_data['LSmallToe'].values) / 2 + joint_data['LHeel'].values) / 2

    # 计算右脚中点
    Mid_r_foot = ((joint_data['RBigToe'].values + joint_data['RSmallToe'].values) / 2 + joint_data['RHeel'].values) / 2

    # 两脚中点作为原点
    origin = np.mean((Mid_r_foot[10:30, :] + Mid_l_foot[10:30, :]) / 2, axis=0)

    # 计算X轴：从右脚指向左脚
    X_axis = np.mean(Mid_l_foot[10:30, :] - Mid_r_foot[10:30, :], axis=0)
    X_axis = X_axis / np.linalg.norm(X_axis)  # 规范化X轴

    # 使X轴与地面平行，正交化X轴（Gram-Schmidt过程）
    Y_axis_1 = np.array([0, 1, 0])
    X_axis = X_axis - np.dot(X_axis, Y_axis_1) * Y_axis_1
    X_axis = X_axis / np.linalg.norm(X_axis)

    # 计算Y轴：从脚中点指向髋中点
    Y_axis = np.mean(joint_data['midHip'].values[10:30, :], axis=0) - origin
    Y_axis = Y_axis / np.linalg.norm(Y_axis)  # 规范化Y轴

    # 正交化Y轴（Gram-Schmidt过程）
    Y_axis = Y_axis - np.dot(Y_axis, X_axis) * X_axis
    Y_axis = Y_axis / np.linalg.norm(Y_axis)

    # 计算Z轴：使用右手法则计算X轴和Y轴的叉积
    Z_axis = np.cross(X_axis, Y_axis)
    Z_axis = Z_axis / np.linalg.norm(Z_axis)  # 规范化Z轴

    # 构造旋转矩阵
    R = np.vstack([X_axis, Y_axis, Z_axis])

    # 预分配变换坐标数组
    transformed_mass = np.zeros(Total_mass.shape)

    # 对每一帧进行坐标变换
    for i in range(Total_mass.shape[0]):
        # 平移
        translated = Total_mass[i, :] - origin

        # 旋转
        transformed_mass[i, :] = np.dot(R, translated.T).T

       # 将变换后的质量数据转换为DataFrame，手动指定列名
    columns_transformed = ['左右', '高低', '前后'] * (transformed_mass.shape[1] // 3)
    
    transformed_mass_df = pd.DataFrame(transformed_mass, columns=columns_transformed)
    
    # 提取第20到第40行数据的平均值作为重心的初始位置
    initial_position = transformed_mass_df.iloc[19:40].mean()
    
    # 计算各个方向的最大位移值，减去重心的初始位置，并求其绝对值
    max_left = abs(transformed_mass_df['左右'].max() - initial_position['左右'])    # 最大左位移
    max_right = abs(transformed_mass_df['左右'].min() - initial_position['左右'])   # 最大右位移
    max_forward = abs(transformed_mass_df['前后'].max() - initial_position['前后']) # 最大前位移
    max_backward = abs(transformed_mass_df['前后'].min() - initial_position['前后'])# 最大后位移
    
    # 计算四个方向的最大位移绝对值
    max_displacement = max(max_left, max_right, max_forward, max_backward)    
    
    # 准备数据
    categories = ['Rightside', 'Forward', 'leftside', 'Backward']
    values = [abs(max_right), abs(max_forward), max_left, abs(max_backward)]
    
    # 创建绘图区域
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # 角度计算
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    
    # 将角度和数据扩展，以闭合图形
    angles += angles[:1]
    values += values[:1]
    
    # 绘制雷达图
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)

    
    # 设置极轴标签
    ax.set_ylim(0, max_displacement)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
        
    # 在顶点位置绘制小圆圈
    for i, value in enumerate(values[:-1]):
        angle_rad = angles[i]
        ax.scatter(angle_rad, value, color='red', s=50)  # s参数控制圆圈的大小

    # 显示各个方向的最大位移值
    for i, value in enumerate(values[:-1]):
        angle_rad = angles[i]
        ax.text(angle_rad, value -0.01, f"{value:.2f} m", ha='center', va='center')
  
    # 叠加足印图像
    footprint_img = plt.imread('balance_analysis/footprint.png')  # 替换为你的足印图像路径
    imagebox = OffsetImage(footprint_img, zoom=0.3, alpha=0.5)  # 调整zoom和alpha参数
    ab = AnnotationBbox(imagebox, (0.5, 0.55), frameon=False, xycoords='axes fraction')
    ax.add_artist(ab)
      
    # 在 Streamlit 中展示图像
    st.pyplot(fig)
    
    
    st.markdown(
        """
        <div style="border: 2px solid green; padding: 10px; border-radius: 10px;">
            <h3>相对于初始重心的各方向最大位移信息</h3>
            <div style="display: flex; justify-content: space-around;">
                <div>向前最大位移:<br>{:.4f} m</div>
                <div>向后最大位移:<br>{:.4f} m</div>
            </div>
            <div style="display: flex; justify-content: space-around;">
                <div>向左最大位移:<br>{:.4f} m</div>
                <div>向右最大位移:<br>{:.4f} m</div>
            </div>
        </div>
        """.format(max_forward, abs(max_backward), max_left, abs(max_right)),
        unsafe_allow_html=True
    )
    
    # 准备三维散点图数据
    scatter_data = go.Scatter3d(
        x=transformed_mass_df['左右'],
        y=transformed_mass_df['高低'],
        z=transformed_mass_df['前后'],
        mode='markers',
        marker=dict(size=3, color='blue'),
        text=["高低: {:.4f} m<br>左右: {:.4f} m<br>前后: {:.4f} m".format(row['高低'], row['左右'], row['前后']) for index, row in transformed_mass_df.iterrows()],
        hoverinfo='text'
    )




    scatter_layout = go.Layout(
        title='重心位移三维散点图',
        scene=dict(
            xaxis=dict(title='左右 (m)', range=[transformed_mass_df['左右'].min(), transformed_mass_df['左右'].max()]),
            yaxis=dict(title='高低 (m)', range=[transformed_mass_df['高低'].min(), transformed_mass_df['高低'].max()]),
            zaxis=dict(title='前后 (m)', range=[transformed_mass_df['前后'].min(), transformed_mass_df['前后'].max()]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=0, y=2, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    scatter_fig = go.Figure(data=[scatter_data], layout=scatter_layout)

    # 在 Streamlit 中展示三维散点图
    st.plotly_chart(scatter_fig)
    st.write('该图像可通过鼠标调整观察角度，该坐标的原点定义是双足中点为空间原点，每一个散点对应的是该时刻重心在该坐标系下的位置，对于数值正负的定义是“相对原点在前数值为正，在后数值为负；在左数值为正，在右数值为负；在上数值为正，在下数值为负”。')
    # 显示最大位移信息

