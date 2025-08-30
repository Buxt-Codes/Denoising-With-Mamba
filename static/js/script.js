const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false
        }
    }
};
const qualityCtx = document.getElementById('qualityChart').getContext('2d');
new Chart(qualityCtx, {
    type: 'doughnut',
    data: {
        labels: ['High Quality', 'Medium Quality', 'Low Quality'],
        datasets: [{
            data: [6850, 4200, 1797],
            backgroundColor: ['#2ecc71', '#f39c12', '#e74c3c'],
            borderWidth: 3,
            borderColor: '#fff'
        }]
    },
    options: {
        ...chartConfig,
        cutout: '60%'
    }
});
const policyCtx = document.getElementById('policyChart').getContext('2d');
new Chart(policyCtx, {
    type: 'bar',
    data: {
        labels: ['Spam', 'Ads', 'Irrelevant', 'Rants', 'Fake'],
        datasets: [{
            data: [324, 189, 156, 98, 80],
            backgroundColor: ['#1e3c72', '#2a5298', '#4a6fa5', '#6b8dd6', '#8bb0ff'],
            borderRadius: 8,
            borderSkipped: false
        }]
    },
    options: {
        ...chartConfig,
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0,0,0,0.05)'
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});
const relevancyCtx = document.getElementById('relevancyChart').getContext('2d');
new Chart(relevancyCtx, {
    type: 'line',
    data: {
        labels: ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'],
        datasets: [{
            label: 'Number of Reviews',
            data: [234, 567, 1890, 4320, 5836],
            borderColor: '#2a5298',
            backgroundColor: 'rgba(42, 82, 152, 0.1)',
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#2a5298',
            pointRadius: 6
        }]
    },
    options: {
        ...chartConfig,
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0,0,0,0.05)'
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});
const temporalCtx = document.getElementById('temporalChart').getContext('2d');
new Chart(temporalCtx, {
    type: 'line',
    data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [
            {
                label: 'High Quality',
                data: [920, 1150, 1080, 1200, 1340, 1420],
                borderColor: '#2ecc71',
                tension: 0.4,
                pointRadius: 4
            },
            {
                label: 'Spam Detected',
                data: [45, 62, 38, 71, 89, 95],
                borderColor: '#e74c3c',
                tension: 0.4,
                pointRadius: 4
            }
        ]
    },
    options: {
        ...chartConfig,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0,0,0,0.05)'
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});
const sampleReviews = [
    {
        text: "Great restaurant with amazing food and friendly staff! The pasta was delicious and the service was excellent. Highly recommend!",
        quality: 92,
        relevancy: 96,
        spam: 3,
        violations: []
    },
    {
        text: "BEST DEALS ON ELECTRONICS! Visit our store now for huge discounts! Call 555-0123",
        quality: 15,
        relevancy: 22,
        spam: 89,
        violations: ['Advertisement', 'Promotional Content']
    },
    {
        text: "This place is terrible and I hate everything about my life. The government is controlling our minds through food additives.",
        quality: 28,
        relevancy: 45,
        spam: 67,
        violations: ['Rant', 'Irrelevant Content']
    },
    {
        text: "The coffee here is okay, nothing special. Service was a bit slow but the atmosphere is nice.",
        quality: 75,
        relevancy: 88,
        spam: 8,
        violations: []
    }
];
function analyzeReview() {
    const reviewText = document.getElementById('reviewText').value.trim();
    const locationType = document.getElementById('locationType').value;
    const reviewerHistory = document.getElementById('reviewerHistory').value;
    if (!reviewText) {
        alert('Please enter a review to analyze.');
        return;
    }
    const resultsSection = document.getElementById('resultsSection');
    const policySection = document.getElementById('policyViolations');
    resultsSection.style.display = 'grid';
    document.getElementById('qualityScore').textContent = '...';
    document.getElementById('relevancyScore').textContent = '...';
    document.getElementById('spamProbability').textContent = '...';
    setTimeout(() => {
        var req = fetch('/model/single', {
            method: 'post',
            body: JSON.stringify({ "text": reviewText, "location": locationType }),
            headers: {
                "Content-Type": "application/json",
            },
        })
        req.then(function (response) {
            
            if (res.ok) {
                results = JSON.parse(response.json());
                displayBatchResults(results);
            } 
        }, function (error) {
            console.error('failed due to network error or cross domain')
        })
    }, 1500);
}
// function simulateMLAnalysis(text, locationType, reviewerHistory) {
//     const spamKeywords = ['best deals', 'call now', 'huge discounts', 'visit our store', 'click here'];
//     const adKeywords = ['buy now', 'sale', 'discount', 'offer', 'promotion', 'website', 'phone number'];
//     const rantKeywords = ['hate', 'terrible', 'worst', 'government', 'conspiracy'];
//     const qualityKeywords = ['great', 'excellent', 'amazing', 'delicious', 'wonderful', 'perfect', 'love'];
//     const relevantKeywords = {
//         restaurant: ['food', 'meal', 'service', 'waiter', 'menu', 'taste', 'delicious'],
//         hotel: ['room', 'bed', 'service', 'clean', 'comfortable', 'staff'],
//         retail: ['product', 'price', 'quality', 'staff', 'service', 'selection'],
//         service: ['service', 'staff', 'professional', 'helpful', 'quality'],
//         entertainment: ['fun', 'entertainment', 'show', 'experience', 'enjoyable']
//     };
//     let qualityScore = 50;
//     let relevancyScore = 50;
//     let spamProbability = 10;
//     let violations = [];
//     const lowerText = text.toLowerCase();
//     spamKeywords.forEach(keyword => {
//         if (lowerText.includes(keyword)) {
//             spamProbability += 25;
//             qualityScore -= 20;
//         }
//     });
//     adKeywords.forEach(keyword => {
//         if (lowerText.includes(keyword)) {
//             violations.push('Advertisement');
//             qualityScore -= 15;
//             relevancyScore -= 10;
//         }
//     });
//     rantKeywords.forEach(keyword => {
//         if (lowerText.includes(keyword)) {
//             violations.push('Rant');
//             qualityScore -= 10;
//         }
//     });
//     qualityKeywords.forEach(keyword => {
//         if (lowerText.includes(keyword)) {
//             qualityScore += 10;
//             relevancyScore += 5;
//         }
//     });
//     const relevantWords = relevantKeywords[locationType] || [];
//     let relevantWordCount = 0;
//     relevantWords.forEach(keyword => {
//         if (lowerText.includes(keyword)) {
//             relevantWordCount++;
//             relevancyScore += 8;
//         }
//     });
//     if (relevantWordCount === 0) {
//         violations.push('Irrelevant Content');
//         relevancyScore -= 20;
//     }
//     if (reviewerHistory === 'verified') {
//         qualityScore += 5;
//         spamProbability -= 10;
//     } else if (reviewerHistory === 'new') {
//         spamProbability += 10;
//     }
//     if (text.length < 10) {
//         violations.push('Insufficient Content');
//         qualityScore -= 20;
//     }
//     if (/(.)\1{4,}/.test(text)) {
//         violations.push('Spam Pattern');
//         spamProbability += 30;
//     }
//     qualityScore = Math.max(0, Math.min(100, qualityScore));
//     relevancyScore = Math.max(0, Math.min(100, relevancyScore));
//     spamProbability = Math.max(0, Math.min(100, spamProbability));
//     violations = [...new Set(violations)];
//     return {
//         quality: qualityScore,
//         relevancy: relevancyScore,
//         spam: spamProbability,
//         violations: violations
//     };
// }
function displayResults(result) {
    const qualityElement = document.getElementById('qualityScore');
    const relevancyElement = document.getElementById('relevancyScore');
    const spamElement = document.getElementById('spamProbability');
    const violationsElement = document.getElementById('policyViolations');
    qualityElement.textContent = result.quality + '%';
    qualityElement.className = 'result-value ' + getQualityClass(result.quality);
    relevancyElement.textContent = result.relevancy + '%';
    relevancyElement.className = 'result-value ' + getQualityClass(result.relevancy);
    spamElement.textContent = result.spam + '%';
    spamElement.className = 'result-value ' + getSpamClass(result.spam);
    if (result.violations.length > 0) {
        violationsElement.innerHTML = result.violations
            .map(violation => `<div class="violation-tag">${violation}</div>`)
            .join('');
        violationsElement.style.display = 'flex';
    } else {
        violationsElement.style.display = 'none';
    }
    animateValue(qualityElement, 0, result.quality, 1000);
    animateValue(relevancyElement, 0, result.relevancy, 1000);
    animateValue(spamElement, 0, result.spam, 1000);
}
function getQualityClass(score) {
    if (score >= 70) return 'quality-high';
    if (score >= 40) return 'quality-medium';
    return 'quality-low';
}
function getSpamClass(score) {
    if (score >= 50) return 'quality-low';
    if (score >= 20) return 'quality-medium';
    return 'quality-high';
}
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const suffix = element.textContent.includes('%') ? '%' : '';
    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = Math.round(start + (end - start) * progress);
        element.textContent = current + suffix;
        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }
    requestAnimationFrame(updateValue);
}
function loadSampleReview(index) {
    if (index < sampleReviews.length) {
        document.getElementById('reviewText').value = sampleReviews[index].text;
    }
}
function updateStatistics() {
    const totalEl = document.getElementById('totalReviews');
    const spamEl = document.getElementById('spamDetected');
    const violationsEl = document.getElementById('policyViolations');
    const currentTotal = parseInt(totalEl.textContent.replace(',', ''));
    const newTotal = currentTotal + Math.floor(Math.random() * 5);
    totalEl.textContent = newTotal.toLocaleString();
    spamEl.textContent = Math.floor(newTotal * 0.1).toLocaleString();
    violationsEl.textContent = Math.floor(newTotal * 0.066).toLocaleString();
}
setInterval(updateStatistics, 30000);
document.addEventListener('DOMContentLoaded', function () {
    const textarea = document.getElementById('reviewText');
    const hints = [
        "Try: 'Great restaurant with amazing food and friendly staff!'",
        "Try: 'BEST DEALS! Visit our website for discounts!'",
        "Try: 'This place is terrible, I hate everything!'",
        "Try: 'Nice hotel, clean rooms and good service'"
    ];
    let hintIndex = 0;
    setInterval(() => {
        if (textarea.value === '') {
            textarea.placeholder = hints[hintIndex];
            hintIndex = (hintIndex + 1) % hints.length;
        }
    }, 4000);
});
let uploadedFiles = [];
let currentInputMethod = 'text';
function switchInputMethod(method) {
    currentInputMethod = method;
    document.querySelectorAll('.input-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    document.querySelectorAll('.input-content').forEach(content => {
        content.classList.remove('active');
    });
    if (method === 'text') {
        document.getElementById('textInput').classList.add('active');
        document.getElementById('analyzeButtonText').textContent = 'Analyze Review';
    } else {
        document.getElementById('fileInput').classList.add('active');
        document.getElementById('analyzeButtonText').textContent = 'Analyze Files';
    }
}
function triggerFileUpload() {
    document.getElementById('fileInputElement').click();
}
function handleDragOver(e) {
    e.preventDefault();
    e.target.closest('.file-upload-area').classList.add('dragover');
}
function handleDragLeave(e) {
    e.preventDefault();
    e.target.closest('.file-upload-area').classList.remove('dragover');
}
function handleDrop(e) {
    e.preventDefault();
    const uploadArea = e.target.closest('.file-upload-area');
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}
function processFiles(files) {
    const validExtensions = ['.csv', '.txt', '.json'];
    const validFiles = files.filter(file => {
        const extension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        return validExtensions.includes(extension);
    });
    if (validFiles.length === 0) {
        alert('Please select valid files (.csv, .txt, .json)');
        return;
    }
    validFiles.forEach(file => {
        if (!uploadedFiles.find(f => f.name === file.name)) {
            uploadedFiles.push(file);
        }
    });
    displayUploadedFiles();
}
function displayUploadedFiles() {
    const container = document.getElementById('uploadedFiles');
    if (uploadedFiles.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';
    container.innerHTML = uploadedFiles.map((file, index) => `
                <div class="file-item">
                    <div class="file-info">
                        <div class="file-icon">${getFileIcon(file.name)}</div>
                        <div class="file-details">
                            <div class="file-name">${file.name}</div>
                            <div class="file-size">${formatFileSize(file.size)}</div>
                        </div>
                    </div>
                    <button class="remove-file" onclick="removeFile(${index})">Remove</button>
                </div>
            `).join('');
}
function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayUploadedFiles();
}
function csvJSON(csv) {

    var lines = csv.split("\n");

    var result = [];

    var headers = lines[0].split(",");

    for (var i = 1; i < lines.length; i++) {

        var obj = {};
        var currentline = lines[i].split(",");

        for (var j = 0; j < headers.length; j++) {
            obj[headers[j]] = currentline[j];
        }

        result.push(obj);

    }

    //return result; //JavaScript object
    return JSON.stringify(result); //JSON
}
function getFileIcon(filename) {
    const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
    switch (extension) {
        case '.csv': return '📊';
        case '.txt': return '📄';
        case '.json': return '📋';
        default: return '📁';
    }
}
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
async function processFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const content = e.target.result;
            let reviews = [];
            try {
                if (file.name.endsWith('.json')) {
                    const data = JSON.parse(content);
                    reviews = Array.isArray(data) ? data : [data];
                } else if (file.name.endsWith('.csv')) {
                    const lines = content.split('\n').filter(line => line.trim());
                    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
                    const reviewIndex = headers.findIndex(h => h.includes('review') || h.includes('text') || h.includes('comment'));
                    if (reviewIndex === -1) {
                        reject(new Error('No review column found in CSV'));
                        return;
                    }
                    reviews = lines.slice(1).map(line => {
                        const values = line.split(',');
                        return {
                            text: values[reviewIndex]?.replace(/"/g, '') || '',
                            location: values[reviewIndex + 1]?.replace(/"/g, '') || ''
                        };
                    }).filter(review => review.text.trim());
                } else if (file.name.endsWith('.txt')) {
                    reviews = content.split('\n')
                        .filter(line => line.trim())
                        .map(text => ({ text: text.trim(), filename: file.name }));
                }
                resolve(reviews);

            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = reject;
        reader.readAsText(file);
    });
}
async function analyzeBatch() {
    const batchSection = document.getElementById('batchAnalysis');
    const progressFill = document.getElementById('progressFill');
    const batchResults = document.getElementById('batchResults');
    batchSection.style.display = 'block';
    let allReviews = [];
    var reviews;
    try {
        reviews = await processFileContent(uploadedFiles[0]);
    } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
    }


    var req = fetch('/model/batch', {
        method: 'post',
        body: JSON.stringify(reviews),
        headers: {
            "Content-Type": "application/json",
        },
    }); // returns a promise

    req.then(function (response) {
        
        if (res.ok) {
            results = JSON.parse(response.json());
            displayBatchResults(results);
        } else {
            // status was something else
        }
    }, function (error) {
        console.error('failed due to network error or cross domain')
    })

}
function displayBatchResults(results) {
    const batchResults = document.getElementById('batchResults');
    const summary = {
        total: results.length,
        highQuality: results.filter(r => r.cf >= 0.7).length,
        spam: results.filter(r => r.label == 1).length
    };
    batchResults.innerHTML = `
                <div style="margin-top: 20px; padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
                    <h3 style="margin-bottom: 15px; color: white;">Batch Analysis Results</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold;">${summary.total}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Total Reviews</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #2ecc71;">${summary.highQuality}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">High Quality</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #e74c3c;">${summary.spam}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Potential Spam</div>
                        </div>
                    </div>
                    <div style="max-height: 300px; overflow-y: auto;">
                        ${results.slice(0, 20).map(result => `
                            <div style="margin-bottom: 10px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 5px; font-size: 0.85rem;">
                                <div style="margin-bottom: 5px;"><strong>Text:</strong> ${result.text}</div>
                                <div style="display: flex; gap: 15px;">
                                    <span>Quality: <strong style="color: ${getQualityColor(result.quality)}">${result.quality}%</strong></span>
                                    <span>Spam: <strong style="color: ${getSpamColor(result.spam)}">${result.spam}%</strong></span>
                                </div>
                            </div>
                        `).join('')}
                        ${results.length > 20 ? `<div style="text-align: center; padding: 10px; opacity: 0.7;">... and ${results.length - 20} more results</div>` : ''}
                    </div>
                </div>
            `;
}
function getQualityColor(score) {
    if (score >= 0.70) return '#2ecc71';
    if (score >= 0.4) return '#f39c12';
    return '#e74c3c';
}
function getSpamColor(score) {
    if (score == 0) return '#e74c3c';
    return '#2ecc71';
}
function analyzeContent() {
    if (currentInputMethod === 'text') {
        analyzeReview();
    } else {
        if (uploadedFiles.length === 0) {
            alert('Please upload files first.');
            return;
        }
        analyzeBatch();
    }
}