<h1>Реализовать модель машинного обучения в production (в виде веб-сервиса, работающего по API)</h1>
<p>Для запуска проекта понадобятся пакеты nuget:</p>
<li>EntityFramework(https://www.nuget.org/packages/EntityFramework/6.4.4)</li>
<li>Microsoft.ML(https://www.nuget.org/packages/Microsoft.ML/1.6.0)</li>
<br>
<p>В качестве модели использовал модель бинарной классификации тональности текста, обученную при выполнении другого задания.<p>
<br>
<p>Для отправки post запроса необходимо открыть powershell, и выполнить команду [Invoke-RestMethod "https://localhost:44303/" -Method Post -Body (@{SentimentText="This was a very bad steak"} | ConvertTo-Json) -ContentType "application/json"], где SentimentText - изменяемое значение, проверяемого комментария.</p>
<p>Так же можно отправить форму по адресу (https://localhost:44303/Predict)</p>
