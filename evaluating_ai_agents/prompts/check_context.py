from langchain_core.prompts import PromptTemplate

check_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template="""\
        Você é um assistente muito prestativo em identificar se uma mensagem apresenta relação com a anterior. 
        A sua tarefa é analisar se a mensagem atual possui relação com a mensagem anterior. 
        Para realizar a sua tarefa considere as regras <instrucao> e <exemplos>

        <instrucao>
        Analise e verifique se <message> possui uma relação de dependência semântica ou contextual com <early_messages>. 
        Se a <message> possuir relação de dependência semântica ou contextual com <early_messages>, apenas responda com um `Sim`. 
        Se a <message> não possuir relação de dependência semântica ou contextual com <early_messages>, apenas responda com um `Não`.
        </instrucao>

        <exemplos>
        Considere os seguintes exemplos para conseguir identificar quando uma mensagem atual apresentar dependência semântica ou contextual com a anterior. 
        <exemplo1> **Mensagem atual**: para construção de chatbots. *Mensagem anterior*: LLM's. **Response model**: Poderia melhor me especificar sobre o que sobre o que gostaria de saber das LLM's?. **Reasoning**: As mensagens aparentam possuir relação entre si. A primeira mensagem é vaga, mas depois da pergunta do modelo, o usuário especifica a sua demanda, utilizando-se de uma preposição, que denota a ligação da mensagem anterior com a atual. Portanto, a minha resposta deve ser essa a seguir: 'Sim'. </exemplo1>
        <exemplo2> **Mensagem atual**: E o que motiva o emprego de tais modelos para essas áreas?. **Response model**:  Sim, as LLM's conhecidas também como modelos de fundação podem ser utilizadas em diversas áreas, tais como no da saúde, em especial, na saúde mental, como perguntou. O seu emprego ocorre, pois apresentam a capacidade de entender a mensagem do usuário e de respondê-la em função de seu pré-treinamento ou de informações adicionais que podem ser provisionadas para o modelo, para a geração de respostas mais corretas, completas e acuradas. Sobre o que mais eu poderia te ajudar?. **Mensagem anterior**: Eu posso utilizar as LLM's para a saúde mental?. **Reasoning**: Analisando a mensagem atual em relação com a anterior, percebe-se que há uma clara ligação de uma com a outra, evidenciada por meio da continuação subjacente. Ao mesmo tempo, a resposta do modelo ajuda a fundamentar a mensagem atual, evidenciando a conexão entre elas existente. Portanto, a minha resposta deve ser a seguinte: 'Sim'. </exemplo2>
        <exemplo3> **Mensagem atual**: Sim, me diz ai. **Response model**: Além disso, vericou-se que os chatbots podem auxiliar na avaliação psicológica e na intervenção do psicoterapeuta. Gostaria de saber sobre? **Mensagem anterior**: Como os chatbots podem ser utilizados na psicologia? **Reasoning** A mensagem anterior parece estar conectada com a atual, uma vez que representa relação de continuidade, motivada pela resposta do modelo. Portanto, a minha resposta deve ser: Sim
        <exemplo4> **Mensagem atual**: O artigo também explora temas de Ux ?. **Response model**: As LLM's devido às suas características de entendimento semântico e de geração de texto podem ser utilizadas para elaborar chatbots especializados para cada cenário, bastando a utilização das técnicas necessárias, que podem ser desde a engenharia de prompt, RAG, fine tuning ou o pré-treinamento de fato. Há algo mais que eu possa te ajudar? **Mensagem anterior**: De forma breve, por que tais modelos podem ser utilizados como chatbots?. **Reasoning**: A mensagem atual não apresenta relação com a anterior, uma vez que se dirigem a dois cenários diferentes e não apresentam nenhuma relação subjacente entre si. Ao mesmo tempo, não há nada na resposta do modelo que possa se conectar com a mensagem atual, evidenciando, novamente, que o usuário está perguntando sobre um novo assunto para o modelo. Portanto, a minha resposta deve ser a seguinte: 'Não'. </exemplo3>
        </exemplos>

        <variaveis>
        <message>: {question}
        <early_messages>: {chat_history}
        </variaveis>

        <resposta>
        Sua resposta:
        </resposta>
"""
)