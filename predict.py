import joblib
import pandas as pd
from sklearn.svm import SVC

X_train_path="X_train_data.csv"
y_train_path="y_train_data.csv"
model_path="/best_svm_model.pkl"


X_colum=['1、您的性别', '2、年级', '3、学科方向', '5、在校期间的生活费', '6、生活费来源', '7、在校期间的平均月消费',
       '8、您通常参与（或愿意参加）哪些校园生活活动？(学术研究或项目)', '8、(社团或学生组织活动)', '8、(体育活动)',
       '8、(艺术或表演活动)', '8、(志愿服务)', '8、(聚会或聚餐)', '8、(其他)',
       '9、您通常参与什么娱乐活动？(观看电影或戏剧演出)', '9、(参加博物馆或艺术展览)', '9、(参加音乐会或演唱会)',
       '9、(餐厅或咖啡馆聚餐)', '9、(旅游或短途旅行)', '9、(观看体育赛事或参加体育活动)', '9、(购物或逛街)',
       '9、(其他)', '10、您通常习惯吃哪些类型的食物？(中餐（如川菜、粤菜等）)', '10、(西餐（如意面等）)',
       '10、(日韩料理（如寿司、烤肉等）)', '10、(快餐或小吃（如盖浇饭、炒面等）)', '10、(素食或健康食品)',
       '10、(甜点或冰淇淋)', '10、(烧烤或火锅)', '10、(其他)', '11、您通常使用哪些内容或问答平台app(知乎)',
       '11、(bilibili)', '11、(微博)', '11、(小红书)', '11、(豆瓣)', '11、(抖音)', '11、(快手)',
       '11、(百度贴吧)', '11、(虎扑)', '11、(其他)', '12、您平常是否使用游戏作为日常休闲活动',
       '13、您通常喜欢玩哪些类型的游戏？(竞速类（例如极品飞车、地平线等）)', '13、(动作类（例如拳皇等）)',
       '13、(射击类（例如CF、CSGO等）)', '13、(角色扮演类（例如巫师、最终幻想等）)', '13、(即时战略类（例如星际争霸等）)',
       '13、(MOBA（例如LOL、王者荣耀等）)', '13、(策略类（例如文明、钢铁雄心等）)', '13、(冒险类（例如古墓丽影等）)',
       '13、(运动类（例如FIFA、实况足球等）)', '13、(音乐类（例如节奏大师等）)', '13、(模拟经营类（例如模拟人生等）)',
       '13、(桌游类（例如三国杀等）)', '13、(休闲游戏（例如胡闹厨房等）)', '13、(其他)',
       '14、您通常使用什么设备进行游戏活动？(手机)', '14、(电脑PC)', '14、(Switch等主机)',
       '15、您平均一个月通常花费多少在游戏中？', '16、您在上课时间会不会选择玩游戏？', '18、您平均每天花费多少时间在游戏上？']
def update_and_train_model(new_data_point, new_class, model_path, X_train_path, y_train_path):
    # Load the training data from files
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)


    # Append the new data point
    X_train = X_train.append(new_data_point, ignore_index=True)
    new_class_df = pd.DataFrame([new_class], columns=["Class"])
    y_train = y_train.append(new_class_df, ignore_index=True)


    # Load the model if it exists
    try:
        model = joblib.load(model_path)
    except:
        model = SVC(probability=True)

    # Train the model
    model.fit(X_train, y_train["Class"])

    # Save the updated training data back to the files
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)

    # Save the updated model
    joblib.dump(model, model_path)

    print("Model updated and saved successfully!")


while(1):
    user_input = input("Enter a number (Enter 1 to exit): ")
    if user_input == "1":
        print("Exiting the loop.")
        break
    print("请根据以下问题输入您的答案（输入序号即可）\n")
    questions = [
    # List of questions from the cleaned dataset
    "你的性别是（1.男 2.女）",
    "你的年纪是（1.大一 2.大二 3.大三 4.大四 5.研究生 6.其他）",
    "学科方向是（1.文科 2.工科 3.理科 4.艺术体育）",
    "在校期间的生活费是（1.1000以下 2.1000-2000 3.2000-3000 4.3000以上）",
    "生活费来源是（1.全部来自家庭 2.部分来自家庭，部分靠自己赚取 3.全部靠自己赚取）",
    "在校期间的平均月消费（1.1000以下 2.1000-2000 3.2000-3000 4.3000以上）",
    "您通常参与（或愿意参加）哪些校园生活活动？(学术研究或项目) 1.是 2.否",
    "(体育活动) 1.是 2.否",
    "(社团或学生组织活动) 1.是 2.否",
    "(艺术或表演活动) 1.是 2.否",
    "(志愿服务) 1.是 2.否",
    "(聚会或聚餐) 1.是 2.否",
    "(其他) 1.是 2.否",
    "您通常参与（或愿意参加）哪些娱乐活动？(观看电影或戏剧演出) 1.是 2.否",
    "(参加博物馆或艺术展览) 1.是 2.否",
    "(参加音乐会或演唱会) 1.是 2.否",
    "(餐厅或咖啡馆聚餐) 1.是 2.否",
    "(旅游或短途旅行) 1.是 2.否",
    "(观看体育赛事或参加体育活动) 1.是 2.否",
    "(购物或逛街) 1.是 2.否",
    "(其他) 1.是 2.否",
    "您通常习惯吃哪些类型的食物？(中餐（如川菜、粤菜等）) 1.是 2.否",
    "(西餐（如意面等）) 1.是 2.否",
    "(日韩料理（如寿司、烤肉等）) 1.是 2.否",
    "(快餐或小吃（如盖浇饭、炒面等）) 1.是 2.否",
    "(素食或健康食品) 1.是 2.否",
    "(甜点或冰淇淋) 1.是 2.否",
    "(烧烤或火锅) 1.是 2.否",
    "(其他) 1.是 2.否",
    "您通常使用哪些内容或问答平台app？(知乎) 1.是 2.否",
    "(bilibili) 1.是 2.否",
    "(微博) 1.是 2.否",
    "(小红书) 1.是 2.否",
    "(豆瓣) 1.是 2.否",
    "(抖音) 1.是 2.否",
    "(快手) 1.是 2.否",
    "(百度贴吧) 1.是 2.否",
    "(虎扑) 1.是 2.否",
    "(其他) 1.是 2.否",
    "您平常是否使用游戏作为日常休闲活动 1.是 2.否",
    "您通常喜欢玩哪些类型的游戏？(竞速类（例如极品飞车、地平线等）) 1.是 2.否",
    "(动作类（例如拳皇等）) 1.是 2.否",
    "(射击类（例如CF、CSGO等）) 1.是 2.否",
    "(角色扮演类（例如巫师、最终幻想等）) 1.是 2.否",
    "(即时战略类（例如星际争霸等）) 1.是 2.否",
    "(MOBA（例如LOL、王者荣耀等）) 1.是 2.否",
    "(策略类（例如文明、钢铁雄心等）) 1.是 2.否",
    "(冒险类（例如古墓丽影等）) 1.是 2.否",
    "(运动类（例如FIFA、实况足球等）) 1.是 2.否",
    "(音乐类（例如节奏大师等）) 1.是 2.否",
    "(模拟经营类（例如模拟人生等）) 1.是 2.否",
    "(桌游类（例如三国杀等）) 1.是 2.否",
    "(休闲游戏（例如胡闹厨房等）) 1.是 2.否",
    "(其他) 1.是 2.否",
    "您通常使用什么设备进行游戏活动？(手机) 1.是 2.否",
    "(电脑PC) 1.是 2.否",
    "(Switch等主机) 1.是 2.否",
    "您平均一个月通常花费多少在游戏中？（1.不花钱 2.少于100 3.100-500 4.500-1000 5.1000以上）",
    "您在上课时间会不会选择玩游戏？ 1.是 2.否",
    "您平均每天花费多少时间在游戏上？（1.小于1小时 2.1-2小时 3.2-4小时 4.4小时以上）"
    ]

    answers = []
    T=0
    for question in questions:
        T=T+1
        if(T>40 and answers[39]=="2"):
            answers.append("-3")
        elif(T!= 1 and T!= 2 and T!= 3 and T!= 4 and T!= 5 and T!= 6 and T!= 40 and T!= 58 and T!= 59 and T!= 60):
            answer = input(f"{question}: ")
            if(answer=="2"):
                answers.append("0")
            else:
                answers.append("1")

        else:
            answer = input(f"{question}: ")
            answers.append(answer)


    print(answers)
    # Load the saved SVM model
    loaded_model = joblib.load(model_path)
    # Convert user input to a DataFrame
    user_input_df = pd.DataFrame([list(map(float,answers))], columns=X_colum)

    # Predict the outcome using the loaded model
    predicted_class = loaded_model.predict(user_input_df)

    # Get the probability of each class
    predicted_proba = loaded_model.decision_function(user_input_df)

    if(predicted_class[0]==2):
        print("不愿意", predicted_proba)
    else:
        print("愿意", predicted_proba)
    result = input("请输入您的真实倾向（1.愿意（玩）2.不愿意（不玩））：")
    update_and_train_model(user_input_df, map(float,result), model_path, X_train_path, y_train_path)