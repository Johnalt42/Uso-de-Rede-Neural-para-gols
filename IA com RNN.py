
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


#Essa função irá separar as colunas da tabela em arrays
def abrir_separar(file_path):
    Dados= []
    Tabela= pd.read_csv(file_path)
    for i in Tabela.columns:
        Dados.append(Tabela[i].to_numpy())

    return Dados

#Depois de separarmos cada coluna precisamos contar quantos gols cada time tem em cada rodada

#Função para transformar uma tabela em dataframe do pandas e contar os gols de um time por ano

def contagem_de_gols (times,Jogos, ano_inicio, ano_final):

        df= pd.DataFrame({'Rodada': Jogos, 'Time': times})

        #Iremos estabeleer o periodo da tabela de 2003 até 2024, ou seja quando o numéro mudar de 38 para 1 será um novo ono

        #Calcula o ano em que a função está
        df['MudancadeAnos']= (df['Rodada'].diff() <0).astype(int)
        df['Ano']=df['MudancadeAnos'].cumsum()+ ano_inicio
        df = df[df['Ano'].between(ano_inicio, ano_final)]
        tabela_gols=df.groupby (['Ano','Time']).size(). unstack(fill_value=0)

        return tabela_gols



#("C:\\Users\\Jonathan\\Documents\\campeonato-brasileiro-gols.csv") Use esse endereço para a tabela, substitua pelo endereço de seu pc

Teste=abrir_separar("C:\\Users\\Jonathan\\Documents\\campeonato-brasileiro-gols.csv")
Tabela=contagem_de_gols(Teste[2], Teste[1], 2003, 2023)

#Agora vamos treinar uma rede neural
#Primeiro passo é  carregar os dados na rede, usamos  o anos de 2003 até 2022, e transformaos em long format
#Temos que indexar os anos e Times
Tabela=Tabela.reset_index().rename(columns={'index':'Ano'})
if 'Time' in Tabela.columns:
    Tabela=Tabela.drop(0)
    Tabela=Tabela.set_index('Ano')

Tabela=Tabela.set_index('Ano')
#Feito Isso vamos normalizar usando scaler

scaler=MinMaxScaler()

Tabela_Normal=pd.DataFrame(
    scaler.fit_transform(Tabela),
    columns=Tabela.columns,
    index=Tabela.index
)
print ("O tamanho do shape dos scalers é:", scaler.min_.shape)
#Passo 2 chamamos uma função para fazer sequencias temporais, já que o modelo precisa de mais de uma ano para prever tendencias
    #A rede neural vai pegar os ultimos 3 anos para analisar e dps de receber
def criar_sequencias (data):
        X=[]
        Y=[]
        for time in data.columns:
                gols=data.loc[2003:2023,time].values
                for i in range(2, len(gols)):
                    X.append(gols[i-2:i]) #Janela Temporal DE 2 ANOS
                    Y.append(gols[i])
        return np.array(X), np.array(Y)

sequencias, teste_2024=criar_sequencias(Tabela_Normal)

Tensor_sequencias = torch.FloatTensor(sequencias).unsqueeze_(-1)
Tensor_teste_2024 =torch.FloatTensor(teste_2024)

class RNNGols(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn=nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)
    def forward (self,x):
            output, _ = self.rnn(x)
            return self.linear(output[:,-1,:])




splint_idx = int(0.8* len(sequencias))
X_train, X_val= Tensor_sequencias[:splint_idx], Tensor_sequencias[splint_idx:]
Y_train, Y_val= Tensor_teste_2024[:splint_idx], Tensor_teste_2024[splint_idx:]

X_train_Loader_float=torch.FloatTensor(X_train)
y_train_Loader_float=torch.FloatTensor(Y_train)



train_dataset=TensorDataset(X_train_Loader_float,y_train_Loader_float)
train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True)

epochs=200
learning_rate=0.01
hidden_size=32
input_size=1
num_layers=1

model = RNNGols(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loop de treino
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in train_loader:

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()  # Remove dimensão extra
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Print a cada 10 épocas
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch + 1}/{epochs} | Loss: {train_loss / len(train_loader):.5f}')

ultimos_3_anos=Tabela_Normal.loc[2021:2023].values.T # Pegamos os ultimos três anos na tabela normal e transformamos em tensor para o modelo
X_2024 =torch.FloatTensor(ultimos_3_anos).unsqueeze(-1)

model.eval()
with torch.no_grad():
    pred_2024= model(X_2024)
    pred_2024= pred_2024.numpy().reshape(1,-1) #Para colocar a matriz ou arrays no formato correto
    print("Oshape da preivasao é", pred_2024.shape)
    pred_2024= scaler.inverse_transform(pred_2024)
    pred_2024=pred_2024.flatten()
    print("O shape final é", pred_2024.shape)


resultados=pd.DataFrame({
    'Time': Tabela.columns,
    'Gols_previstos': pred_2024.flatten().round().astype(int)
})
print("\n Previsão de Gols em 2024")
print(resultados)
