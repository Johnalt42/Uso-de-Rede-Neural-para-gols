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



#("C:\\Users\\Jonathan\\Documents\\campeonato-brasileiro-gols.csv") Use esse endereço para a tabela

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

class Previsao(nn.Module):
    def __init__(self,input_size=1,hidden_size=16, num_layers=2):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first=True
        , dropout=0.2)
        self.linear=nn.Linear(hidden_size,1)

    def forward(self, x):
        lstm_out,_=self.lstm(x)
        return self.linear(lstm_out[:,-1,:])


modelo=Previsao()


#Com a rede neural definida fazemos a divisão dos dados 80% treino outros para validação
splint_idx = int(0.8* len(sequencias))
X_train, X_val= Tensor_sequencias[:splint_idx], Tensor_sequencias[splint_idx:]
Y_train, Y_val= Tensor_teste_2024[:splint_idx], Tensor_teste_2024[splint_idx:]

#Criando Data loaders
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=16)

#Configurações de Rede neural

criterion = nn.MSELoss()
otimizador= torch.optim.Adam(modelo.parameters(), lr=0.001)
n_epochs = 100

for epoch in range(n_epochs):
    modelo.train()
    train_loss = 0.0

    for batch_X, batch_Y in train_loader:
        otimizador.zero_grad()
        outputs = modelo(batch_X)
        loss=criterion(outputs, batch_Y)
        loss.backward()
        otimizador.step()
        train_loss+=loss.item()

    # Para validar
    modelo.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            outputs = modelo(batch_X)
            val_loss+=criterion(outputs.squeeze(), batch_Y.unsqueeze(1)).item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Época {epoch + 1}/{n_epochs} | Loss Treino: {train_loss / len(train_loader):.4f} | Loss Val: {val_loss / len(val_loader):.4f}")


#Uma vez feito a rede neural no sistema com perdas e analises vamos tentar prever os gols de 2024 com os ultimos 3 anos

ultimos_3_anos=Tabela_Normal.loc[2021:2023].values.T # Pegamos os ultimos três anos na tabela normal e transformamos em tensor para o modelo
X_2024 =torch.FloatTensor(ultimos_3_anos).unsqueeze(-1)

modelo.eval()
with torch.no_grad():
    pred_2024= modelo(X_2024)
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
