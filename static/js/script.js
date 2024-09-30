function makePrediction() {
    // Get the text and number inputs
    const text = document.getElementById("textInput").value;
    const number = document.getElementById("numberInput").value;

    // Prepare the data to be sent to the Flask backend
    const data = {
        text: text,
        number: Number(number)  // Convert to number
    };

    // Send a POST request to the Flask API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())  // Parse the JSON response
    .then(result => {
        // Display the result in the 'result' div
        const formattedText = result.predicted_class.replace(/\n/g, '<br>');
        console.log(formattedText)
        document.getElementById('result').innerHTML = 
            `Predicted class: ${formattedText}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 
            `Error: ${error.message}`;
    });
}
