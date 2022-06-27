### Deep Neural Network with MLE


![image](https://user-images.githubusercontent.com/86671456/175959202-0d8187f3-d4d7-4ff6-8be9-5beb38745df4.png)


- 분포 P(x)로부터 샘플링한 데이터 x가 주어졌을 때, 파라미터 세타를 갖는 DNN은 조건부 확률 분포를 나타낸다.

![image](https://user-images.githubusercontent.com/86671456/175959797-23a380b0-30e6-481f-a636-3860c4b20a9b.png)

- Gradient Descent를 통해서 NLL(Negative log Likehood)를 최소화하는 파라미터 세타를 찾을 수 있다.(즉 Max->Min로 바뀐 것뿐이다.)

![image](https://user-images.githubusercontent.com/86671456/175960022-a70eada4-ef2f-4e07-bb0e-a80f125cdaa2.png)


<br>

- MLE를 통해 수집한 데이터셋을 잘 설명하는 확률 분포의 파라미터를 찾을 수 있다.(Likehood)

- Neural Networks 또한 확률 분포 함수이므로, MLE를 통해 파라미터 찾기

   - 최대화 대신 최소화를 위해서 Negative Log-Likelihood(NLL)을 Gradient Descent

- Gradient Descent를 수행하기 위한, 파라미터에 대한 미분이 필요( by back-propagation)
