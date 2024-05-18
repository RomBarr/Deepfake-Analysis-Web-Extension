/*
  File: background.js
  Version: 1.0
  Description: background script for a Chrome extension. It contains functions to send image URLs to a Flask application for processing,
  handle responses, store results in Chrome storage, and send notifications to the user.
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
*/

// Background.js
function sendImageUrlToFlask(imageUrl) {
    console.log('Entered sendImageUrlToFlask function');
    fetch('http://localhost:5000/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_url: imageUrl })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to send image URL to Flask application');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            console.error('Server Error:', data.error);
            return;
        }
        console.log('Response from Flask:', data.probability_results);

        // Store the imageURL and results in Chrome storage using the unique key
        var timestamp = Date.now(); //unique timestamp
        var storageData = {};
        storageData[timestamp] = {
            'imageURL': imageUrl,
            'results': data.probability_results
        };

        chrome.storage.local.set(storageData, () => {
            console.log('Image URL and results stored in Chrome storage');
            sendNotification('Image Scan', 'Image processing complete.');
        });

        // Saving most recent result
        var latestResult = {
        'latestResult': {
            'timestamp': timestamp,
            'imageURL': imageUrl,
            'results': data.probability_results
          }
        };

        chrome.storage.local.set(latestResult, () => {
        console.log('Latest result stored temporarily in Chrome storage');
        });
    })
    .catch(error => {
        console.error('Error:', error);
        sendNotification('Image Scan', 'Error processing image.');
    });
}

function sendNotification(title, message) {
    chrome.notifications.create('', {
        title: title,
        message: message,
        iconUrl: 'images/marketing-automation.png',
        type: 'basic'
    });
}

chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    if (message.type === "imageClicked") {
        sendNotification('Image Scan', 'Processing image...');
        sendImageUrlToFlask(message.url);
    } else if (message.type === "imageScanCancelled") {
        sendNotification('Image Scan', 'Image scan cancelled.');
    }
});
