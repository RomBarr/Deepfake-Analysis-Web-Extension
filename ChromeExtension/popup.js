/*
  File: popus.js
  Version: 1.0
  Description: Controls the behavior of the popup window for a Chrome extension.
  It handles user interactions such as initiating image scans, displaying scan history,
  clearing history, and responding to messages from the background script.
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
*/

document.addEventListener("DOMContentLoaded", function() {
  var scanImageButton = document.getElementById("scanImageBtn");
  var bodyElement = document.body;

  scanImageButton.addEventListener("click", function() {
    // Toggle active class on button
    scanImageButton.classList.toggle('button-active');
    bodyElement.classList.toggle('body-image-selecting');

    // Send message to content script to activate listener
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {type: "activateImageScan"});
    });
  });

  // Listen for completion or cancellation
  chrome.runtime.onMessage.addListener(function(message) {
    if (message.type === "imageScanComplete" || message.type === "imageScanCancelled") {
      // Remove active styles if image click is complete or cancelled
      scanImageButton.classList.remove('button-active');
      bodyElement.classList.remove('body-image-selecting');
    }
  });
});

var historyBtn = document.getElementById("historyBtn");
var resultsContainer = document.getElementById("resultsContainer");
var isHistoryVisible = false; // To track visibility of history records

historyBtn.addEventListener("click", function() {
  if (!isHistoryVisible) {
    // Fetch and display history if not currently visible
    chrome.storage.local.get(null, function(result) {
      if (chrome.runtime.lastError) {
        console.error("Error retrieving data:", chrome.runtime.lastError);
      } else {
        resultsContainer.innerHTML = ""; // Clear previous content
        var keys = Object.keys(result).filter(key => key !== 'latestResult').sort(function(a, b) { return b - a; }); // Exclude 'latestResult' and sort
        keys.forEach(function(timestamp) {
          var data = result[timestamp];
          var formattedResults = (parseFloat(data.results)).toFixed(2) + '%'; // Format the result as percentage
          var listItem = document.createElement("div");
          listItem.classList.add("history-item");
          listItem.innerHTML = `<strong>Date/Time:</strong> ${new Date(parseInt(timestamp)).toLocaleString()}<br>
                                <strong>URL:</strong> <a href='${data.imageURL}' target='_blank'>Link</a><br>
                                <strong>DeepFake Probability:</strong> ${formattedResults}`;
          resultsContainer.appendChild(listItem);
        });
        isHistoryVisible = true;
      }
    });
  } else {
    resultsContainer.innerHTML = "";
    isHistoryVisible = false;
  }
});

function displayResult(data) {
    var resultsContainer = document.getElementById('resultsContainer');
    var formattedResults = (parseFloat(data.results)).toFixed(2) + '%'; // Assuming results are numeric
    var listItem = document.createElement("div");

    var label = document.createElement("div");
    label.innerHTML = "<strong>Most Recent Analysis</strong>";
    label.classList.add("recent-result-label");

    listItem.classList.add("history-item");
    listItem.innerHTML = `<strong>Date/Time:</strong> ${new Date(parseInt(data.timestamp)).toLocaleString()}<br>
                          <strong>URL:</strong> <a href='${data.imageURL}' target='_blank'>Link</a><br>
                          <strong>DeepFake Probability:</strong> ${formattedResults}`;
    resultsContainer.innerHTML = ""; // Clear any previous results first
    resultsContainer.appendChild(label);
    resultsContainer.appendChild(listItem);
}

document.addEventListener('DOMContentLoaded', function() {
    chrome.storage.local.get('latestResult', function(result) {
        if (result.latestResult) {
            displayResult(result.latestResult);
        }
    });
});

var clearHistoryBtn = document.getElementById("clearHistoryBtn");
clearHistoryBtn.addEventListener("click", function() {
  chrome.storage.local.clear(function() {
    if (chrome.runtime.lastError) {
      console.error("Error clearing data:", chrome.runtime.lastError);
    } else {
      console.log("All stored data cleared successfully");
      var resultsContainer = document.getElementById("resultsContainer");
      resultsContainer.innerHTML = "History Cleared"; // This will clear the displayed results
    }
  });
});
