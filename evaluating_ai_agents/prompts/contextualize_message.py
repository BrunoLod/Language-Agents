from langchain_core.prompts import PromptTemplate

contextualize_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template="""\
        Você é um assistente muito prestativo em identificar se uma mensagem apresenta relação com a anterior. 
        A sua tarefa é analisar se a mensagem atual possui relação com a mensagem anterior e, se possuir, contextualizá-la com base na outra.
        Para realizar a sua tarefa considere as regras <instrucao> e <exemplos>. 

        <instrucao>
        Analise e verifique se <message> possui uma relação de dependência semântica ou contextual com <early_messages>. 
        Se a <message> possuir relação de dependência semântica ou contextual com <early_messages>, contextualize-a com base nela.
        Se a <message> não possuir relação de dependência semântica ou contextual com <early_messages>, não a contextualize, mas a matenha coforme se apresenta.
        </instrucao>

        <exemplos>
        Considere os seguintes exemplos para conseguir identificar quando uma mensagem atual apresentar dependência semântica ou contextual com a anterior e desempenhar o comportamento esperado. 
        <exemplo1> **Mensagem atual**: para construção de chatbots. *Mensagem anterior*: LLM's. **Response model**: Poderia melhor me especificar sobre o que sobre o que gostaria de saber das LLM's?. **Reasoning**: As mensagens aparentam possuir relação entre si. A primeira mensagem é vaga, mas depois da pergunta do modelo, o usuário especifica a sua demanda, utilizando-se de uma preposição, que denota a ligação da mensagem anterior com a atual. Portanto, preciso contextualizar a mensagem atual com base na anterior, de forma similar a essa: Uso de LLM's para a construção de chatbots. </exemplo1>
        <exemplo2> **Mensagem atual**: E o que motiva o emprego de tais modelos para essa área?. **Response model**:  Sim, as LLM's conhecidas também como modelos de fundação podem ser utilizadas em diversas áreas, tais como no da saúde, em especial, na saúde mental, como perguntou. O seu emprego ocorre, pois apresentam a capacidade de entender a mensagem do usuário e de respondê-la em função de seu pré-treinamento ou de informações adicionais que podem ser provisionadas para o modelo, para a geração de respostas mais corretas, completas e acuradas. Sobre o que mais eu poderia te ajudar?. **Mensagem anterior**: Eu posso utilizar as LLM's para a saúde mental?. **Reasoning**: Analisando a mensagem atual em relação com a anterior, percebe-se que há uma clara ligação de uma com a outra, evidenciada por meio da continuação subjacente. Ao mesmo tempo, a resposta do modelo ajuda a fundamentar a mensagem atual, evidenciando a conexão entre elas existente. Portanto, devo contextualizar a mensagem atual em função da anterior, de forma similar a essa aqui: E o que motiva o emprego dos modelos de LLM para diversas áreas, como a da saúde?. </exemplo2>
        <exemplo3> **Mensagem atual**: O artigo também explora temas de Ux ?. **Response model**: As LLM's devido às suas características de entendimento semântico e de geração de texto podem ser utilizadas para elaborar chatbots especializados para cada cenário, bastando a utilização das técnicas necessárias, que podem ser desde a engenharia de prompt, RAG, fine tuning ou o pré-treinamento de fato. Há algo mais que eu possa te ajudar? **Mensagem anterior**: De forma breve, por que tais modelos podem ser utilizados como chatbots?. **Reasoning**: A mensagem atual não apresenta relação com a anterior, uma vez que se dirigem a dois cenários diferentes e não apresentam nenhuma relação subjacente entre si. Ao mesmo tempo, não há nada na resposta do modelo que possa se conectar com a mensagem atual, evidenciando, novamente, que o usuário está perguntando sobre um novo assunto para o modelo. Portanto, não devo contextualizar a mensagem atual em função da anterior, mas mantê-la como se apresenta: O artigo também explora temas de Ux?. </exemplo3>
        <exemplos>

        <variaveis>
        <message>: {question}
        <early_messages>: {chat_history}
        </variaveis>

        <resposta>
        Sua resposta:
        </resposta>
    """
)