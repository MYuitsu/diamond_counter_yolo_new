//XỬ LÝ TÍNH NĂNG TỰ ĐỘNG ĐẾM KIM CƯƠNG VỚI KHAY
// Tạo một biến toàn cục để lưu trữ vùng khay
window.trayRegion = null;
// Hàm này sẽ được gọi khi người dùng xác nhận vùng khay
// region_x, region_y, region_w, region_h là tọa độ và kích thước của vùng khay
// tray_width, tray_length là kích thước của khay
function startAutoCountWithTray(region_x, region_y, region_w, region_h, tray_width, tray_length) {
    window.trayRegion = { region_x, region_y, region_w, region_h, tray_width, tray_length };
    autoCountDiamonds();
}
// Hàm này sẽ tự động đếm kim cương trong vùng đã chọn
// Nó sẽ gửi yêu cầu đến server để xử lý ảnh và trả về kết quả
async function autoCountDiamonds() {
    setUIEnabled(false);
    try {
        await fetch("/reset", { method: "POST" });
        const canvas = document.getElementById("canvasUpload");
        const partSize = parseInt(document.getElementById("partSizeSelect").value);
        const stride = partSize / 2;

        let region_x = 0, region_y = 0, region_w = canvas.width, region_h = canvas.height;
        if (window.trayRegion) {
            region_x = window.trayRegion.region_x;
            region_y = window.trayRegion.region_y;
            region_w = window.trayRegion.region_w;
            region_h = window.trayRegion.region_h;
            trayLength = window.trayRegion.tray_length;
            trayWidth = window.trayRegion.tray_width;
        }

        const fontSize = parseInt(document.getElementById("fontSize").value);
        const minSize = parseFloat(document.getElementById("minSize").value);
        const circleThreshold = parseFloat(document.getElementById("circleThreshold").value);


        let totalDiamonds = 0;


        // Lấy scale giữa canvas hiển thị (CSS) và canvas nội bộ (width 640)
        const scale = canvas.clientWidth / canvas.width;

        let promises = [];
        for (let y = 0; y <= canvas.height - partSize; y += stride) {
            for (let x = 0; x <= canvas.width - partSize; x += stride) {

                // Hiển thị box trên giao diện theo tỷ lệ CSS
                box.style.left = (x * scale) + "px";
                box.style.top = (y * scale) + "px";
                box.style.width = (partSize * scale) + "px";
                box.style.height = (partSize * scale) + "px";

                await new Promise(resolve => setTimeout(resolve, 100)); // Cập nhật giao diện

                // Cắt toàn ảnh (vì YOLO sẽ xử lý qua tọa độ vùng)
                const promise = (async () => {
                    const canvasBlob = await new Promise(resolve =>
                        canvas.toBlob(resolve, "image/png")
                    );
                    const formData = new FormData();
                    formData.append("image", canvasBlob, `part_${x}_${y}.png`);
                    formData.append("tray_length", trayLength);
                    formData.append("tray_width", trayWidth);
                    formData.append("min_size", minSize);
                    formData.append("font_size", fontSize);
                    formData.append("region_x", x);
                    formData.append("region_y", y);
                    formData.append("region_w", partSize);
                    formData.append("region_h", partSize);
                    formData.append("circle_threshold", circleThreshold);



                    try {
                        const res = await fetch("/predict?mode=box", {
                            method: "POST",
                            body: formData,
                        });

                    } catch (error) {
                        console.error(`Lỗi khi xử lý vùng (${x}, ${y})`, error);
                        return null;
                    }
                })();
                promises.push(promise);
            }
        }
        await Promise.all(promises);
        const formData = new FormData();
        formData.append("region_x", region_x);
        formData.append("region_y", region_y);
        formData.append("region_w", region_w);
        formData.append("region_h", region_h);
        formData.append("tray_width", trayWidth);
        formData.append("tray_length", trayLength);
        formData.append("min_size", minSize);
        formData.append("circle_threshold", circleThreshold);


        // Lấy file ảnh gốc từ input
        const fileInput = document.getElementById("fileInput");
        const originalFile = fileInput.files[0];
        if (originalFile) {
            formData.append("original_image", originalFile, originalFile.name);
        }

        // Gửi ảnh đã xử lý lên server để lấy kết quả cuối cùng
        const res = await fetch("/final_result", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.annotated_image) {
            const img = new Image();
            img.onload = () => originalAnnotatedImageOnload(img);
            img.src = "data:image/jpeg;base64," + data.annotated_image;
        }
        
        document.getElementById("totalCountDisplay").innerText = `Tổng số: ${data.total_diamonds}💎`;

        // Hiển thị bảng chi tiết
        const tbody = document.getElementById("resultsTableBody");
        tbody.innerHTML = "";
        if (Array.isArray(data.details)) {
            data.details.forEach((item) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
      <td style="padding-left: 20px;">${item.index}</td>
      <td style="padding-left: 20px;">${item.width?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.height?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.shape ?? ""}</td>
      <td style="padding-left: 20px;">${item.area?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.diameter?.toFixed(2) ?? ""}</td>
    `;
                tbody.appendChild(tr);
            });
        }
    } finally {
        setUIEnabled(true);
    }
}


  document.getElementById("saveExcelBtn").addEventListener("click", function () {
    // Lấy bảng
    var table = document.querySelector("table");

    // Chuyển bảng HTML thành sheet
    var wb = XLSX.utils.table_to_book(table, { sheet: "Kết quả" });

    // Tạo tên file kèm ngày giờ
    var now = new Date();
    var fileName = "ket_qua_" 
      + now.getFullYear() + "-"
      + String(now.getMonth() + 1).padStart(2, '0') + "-"
      + String(now.getDate()).padStart(2, '0') + "_"
      + String(now.getHours()).padStart(2, '0') + "-"
      + String(now.getMinutes()).padStart(2, '0') + "-"
      + String(now.getSeconds()).padStart(2, '0')
      + ".xlsx";

    // Xuất file
    XLSX.writeFile(wb, fileName);
  });




// Xử lý sự kiện khi người dùng nhấn nút "Tự động cắt khay"
document.getElementById('autoCutBtn').addEventListener('click', function () {
    document.getElementById('cutOptionsGroup').classList.add('show');
    document.getElementById('confirmTrayBtn').style.display = '';
});


// Xử lý sự kiện khi người dùng xác nhận vùng khay
document.getElementById("confirmTrayBtn").addEventListener("click", function () {
    document.getElementById('cutOptionsGroup').classList.remove('show');
    const scaleFactor = TARGET_SIZE / 320;
    const region_x = Math.round(box.offsetLeft * scaleFactor);
    const region_y = Math.round(box.offsetTop * scaleFactor);
    const region_w = Math.round(box.offsetWidth * scaleFactor);
    const region_h = Math.round(box.offsetHeight * scaleFactor);

    const tray_width = parseFloat(document.getElementById("trayWidth").value);
    const tray_length = parseFloat(document.getElementById("trayLength").value);
    const minSize = parseFloat(document.getElementById("minSize").value);

    // Lưu vào biến toàn cục để autoCountDiamonds dùng
    window.trayRegion = { region_x, region_y, region_w, region_h, tray_width, tray_length, minSize };

    // Gọi autoCountDiamonds
    box.style.border = "2px solid red"; // Hồng
    document.getElementById("confirmTrayBtn").style.display = "none";
    autoCountDiamonds();
});

// Hàm này sẽ bật/tắt các nút và làm mờ ảnh preview
function setUIEnabled(enabled) {
    // Danh sách id các nút cần disable/enable
    const ids = [
        "fileInput", "enhanceBtn", "autoCutBtn", "openCameraBtn", "takePhotoBtn",
        "calculateBtn", "btnWithBox", "btnNumberOnly", "partSizeSelect"
    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
    });
    // Làm mờ ảnh review
    const preview = document.getElementById("canvasUpload");
    if (preview) {
        if (!enabled) preview.classList.add("disabled-overlay");
        else preview.classList.remove("disabled-overlay");
    }
}

//XỬ LÝ NÂNG CẤP ẢNH
// Khi người dùng nhấn nút "Nâng cấp ảnh", sẽ làm sắc nét ảnh
// Hàm này sẽ làm sắc nét ảnh bằng bộ lọc
// Sử dụng bộ lọc sắc nét đơn giản
function enhanceImage() {
    const canvas = document.getElementById("canvasUpload");
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    const kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];

    const copy = new Uint8ClampedArray(data); // Bản sao dữ liệu gốc
    const w = canvas.width, h = canvas.height;

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            for (let c = 0; c < 3; c++) {
                let i = (y * w + x) * 4 + c;
                let val = 0;

                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const weight = kernel[(ky + 1) * 3 + (kx + 1)];
                        const ni = ((y + ky) * w + (x + kx)) * 4 + c;
                        val += copy[ni] * weight;
                    }
                }

                data[i] = Math.min(255, Math.max(0, val));
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);

}


// Xử lý sự kiện khi người dùng nhấn nút "Nâng cấp ảnh"
document.getElementById("enhanceBtn").addEventListener("click", enhanceImage);

// Hiển thị giá trị font size và min size khi người dùng thay đổi
document.getElementById("fontSize").addEventListener("input", function () {
    document.getElementById("fontSizeValue").innerText = this.value;
});

// Hiển thị giá trị min size khi người dùng thay đổi
document.getElementById("minSize").addEventListener("input", function () {
    document.getElementById("minSizeValue").innerText = this.value;
});

// Hàm này sẽ resize và pad ảnh về kích thước 640x640
// Nó sẽ giữ nguyên tỷ lệ ảnh gốc và thêm viền đen nếu cần
const TARGET_SIZE = 640;
function resizeAndPadImage(file, targetSize) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
                const ratio = Math.min(targetSize / img.width, targetSize / img.height);
                const newWidth = Math.round(img.width * ratio);
                const newHeight = Math.round(img.height * ratio);

                const canvas = document.getElementById("canvasUpload");
                canvas.width = targetSize;
                canvas.height = targetSize;
                canvas.style.width = "320px";
                const ctx = canvas.getContext("2d");

                ctx.fillStyle = "black";
                ctx.fillRect(0, 0, targetSize, targetSize);

                const xOffset = Math.floor((targetSize - newWidth) / 2);
                const yOffset = Math.floor((targetSize - newHeight) / 2);
                ctx.drawImage(img, xOffset, yOffset, newWidth, newHeight);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, file.type);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}


// Hàm này sẽ gửi ảnh đã chọn lên server để xử lý
// Nó sẽ gửi ảnh đã resize và pad, cùng với các thông tin khác như vùng chọn
async function sendImageToServer(mode) {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("Vui lòng chọn ảnh trước.");
        return;
    }

    const canvas = document.getElementById("canvasUpload");
    const processedBlob = await new Promise(resolve =>
        canvas.toBlob(blob => resolve(blob), file.type)
    );
    const formData = new FormData();

    const scaleFactor = TARGET_SIZE / 320;
    const boxX = Math.round(box.offsetLeft * scaleFactor);
    const boxY = Math.round(box.offsetTop * scaleFactor);
    const boxW = Math.round(box.offsetWidth * scaleFactor);
    const boxH = Math.round(box.offsetHeight * scaleFactor);

    // Gửi thêm thông tin vùng chọn
    formData.append("region_x", boxX);
    formData.append("region_y", boxY);
    formData.append("region_w", boxW);
    formData.append("region_h", boxH);


    const fontSize = parseInt(document.getElementById("fontSize").value);
    formData.append("font_size", fontSize);


    const minSize = parseFloat(document.getElementById("minSize").value);
    formData.append("min_size", minSize);

    const circleThreshold = parseFloat(document.getElementById("circleThreshold").value);
    formData.append("circle_threshold", circleThreshold);



    formData.append("image", processedBlob, file.name);
    const trayLength = document.getElementById("trayLength").value;
    const trayWidth = document.getElementById("trayWidth").value;

    formData.append("tray_length", trayLength);
    formData.append("tray_width", trayWidth);
    try {
        const res = await fetch(`/predict?mode=${mode}`, {
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        document.getElementById("totalCountDisplay").innerText = "Tổng số: " + data.diamond_count + "💎";
        console.log("chieudai_khay + chieurong_khay hihi:", data.chieudai_khay, data.chieurong_khay);
        if (data.annotated_image) {
            const img = new Image();
            img.onload = () => originalAnnotatedImageOnload(img);
            img.src = "data:image/jpeg;base64," + data.annotated_image;
        }

        const tbody = document.getElementById("resultsTableBody");
        tbody.innerHTML = "";
        if (Array.isArray(data.details)) {
            data.details.forEach((item) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
    <td style="padding-left: 20px;">${item.index}</td>
    <td style="padding-left: 20px;">${item.width.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.height.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.shape}</td>
    <td style="padding-left: 20px;">${item.area.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.diameter.toFixed(2)}</td>
  `;
                tbody.appendChild(tr);
            });
        }
    } catch (error) {
        alert("Lỗi khi gửi ảnh lên server.");
        console.error(error);
    }
}


// Xử lý sự kiện khi người dùng chọn ảnh từ máy tính
// Khi người dùng chọn ảnh, sẽ resize và pad ảnh về kích thước 640x640
document.getElementById("fileInput").addEventListener("change", async function () {
    const file = this.files[0];
    if (file) {
        await resizeAndPadImage(file, TARGET_SIZE);
        showResizableBox();
    }
});


// Xử lý sự kiện khi người dùng nhấn nút "Tự động đếm kim cương"
document.getElementById("btnWithBox").addEventListener("click", () => sendImageToServer("box"));


// Hiển thị khung resizable
// Khung này sẽ cho phép người dùng kéo thả để chọn vùng cần đếm kim
let box = document.getElementById("resizableBox");

// Thiết lập canvas để hiển thị ảnh đã chọn
// Canvas này sẽ hiển thị ảnh đã resize và pad, cùng với vùng chọn
let canvas = document.getElementById("canvasUpload");


// Hiển thị khung resizable khi người dùng chọn ảnh
// Hàm này sẽ hiển thị khung resizable và đặt vị trí, kích thước ban đầu
// Khung này sẽ cho phép người dùng kéo thả để chọn vùng cần đếm kim cương
function showResizableBox() {
    box.style.display = "block";
    box.style.left = "20px";
    box.style.top = "20px";
    box.style.width = "100px";
    box.style.height = "100px";
}

let isDragging = false;
let offsetX = 0, offsetY = 0;

// Kéo di chuyển khung bằng cảm ứng
box.addEventListener("touchstart", function (e) {
    if (e.target.classList.contains("resizer")) return;
    isDragging = true;
    const touch = e.touches[0];
    const rect = box.getBoundingClientRect();
    offsetX = touch.clientX - rect.left;
    offsetY = touch.clientY - rect.top;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchmove", function (e) {
    if (!isDragging) return;
    const touch = e.touches[0];
    let previewRect = canvas.getBoundingClientRect();
    let x = touch.clientX - previewRect.left - offsetX;
    let y = touch.clientY - previewRect.top - offsetY;

    x = Math.max(0, Math.min(x, 320 - box.offsetWidth));
    y = Math.max(0, Math.min(y, 320 - box.offsetHeight));

    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchend", () => isDragging = false);

// Resize 4 góc bằng cảm ứng
const resizers = box.querySelectorAll(".resizer");
let isResizing = false;
let currentResizer;

resizers.forEach(resizer => {
    resizer.addEventListener("touchstart", function (e) {
        isResizing = true;
        currentResizer = resizer;
        e.preventDefault();
        e.stopPropagation();
    }, { passive: false });
});

document.addEventListener("touchmove", function (e) {
    if (!isResizing) return;
    const touch = e.touches[0];
    const canvasRect = canvas.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();

    const dx = touch.clientX - boxRect.left;
    const dy = touch.clientY - boxRect.top;

    let width = box.offsetWidth;
    let height = box.offsetHeight;
    let left = box.offsetLeft;
    let top = box.offsetTop;

    if (currentResizer.classList.contains("bottom-right")) {
        width = dx;
        height = dy;
    } else if (currentResizer.classList.contains("bottom-left")) {
        width = box.offsetWidth - (touch.clientX - boxRect.left);
        left += (touch.clientX - boxRect.left);
        height = dy;
    } else if (currentResizer.classList.contains("top-right")) {
        height = box.offsetHeight - (touch.clientY - boxRect.top);
        top += (touch.clientY - boxRect.top);
        width = dx;
    } else if (currentResizer.classList.contains("top-left")) {
        width = box.offsetWidth - (touch.clientX - boxRect.left);
        height = box.offsetHeight - (touch.clientY - boxRect.top);
        left += (touch.clientX - boxRect.left);
        top += (touch.clientY - boxRect.top);
    }

    // Giới hạn để khung không tràn ra ngoài
    left = Math.max(0, Math.min(left, 320 - 20));
    top = Math.max(0, Math.min(top, 320 - 20));
    width = Math.max(20, Math.min(width, 320 - left));
    height = Math.max(20, Math.min(height, 320 - top));

    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchend", () => isResizing = false);

// Resize bằng chuột
resizers.forEach(resizer => {
    resizer.addEventListener("mousedown", function (e) {
        isResizing = true;
        currentResizer = resizer;
        e.preventDefault();
        e.stopPropagation();
    });
});

document.addEventListener("mousemove", function (e) {
    if (!isResizing) return;
    const canvasRect = canvas.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();

    const dx = e.clientX - boxRect.left;
    const dy = e.clientY - boxRect.top;

    let width = box.offsetWidth;
    let height = box.offsetHeight;
    let left = box.offsetLeft;
    let top = box.offsetTop;

    if (currentResizer.classList.contains("bottom-right")) {
        width = dx;
        height = dy;
    } else if (currentResizer.classList.contains("bottom-left")) {
        width = box.offsetWidth - (e.clientX - boxRect.left);
        left += (e.clientX - boxRect.left);
        height = dy;
    } else if (currentResizer.classList.contains("top-right")) {
        height = box.offsetHeight - (e.clientY - boxRect.top);
        top += (e.clientY - boxRect.top);
        width = dx;
    } else if (currentResizer.classList.contains("top-left")) {
        width = box.offsetWidth - (e.clientX - boxRect.left);
        height = box.offsetHeight - (e.clientY - boxRect.top);
        left += (e.clientX - boxRect.left);
        top += (e.clientY - boxRect.top);
    }

    // Giới hạn để khung không tràn ra ngoài
    left = Math.max(0, Math.min(left, 320 - 20));
    top = Math.max(0, Math.min(top, 320 - 20));
    width = Math.max(20, Math.min(width, 320 - left));
    height = Math.max(20, Math.min(height, 320 - top));

    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
});

document.addEventListener("mouseup", () => isResizing = false);

// XỬ LÝ MỞ CAMERA VÀ CHỤP ẢNH
// Biến toàn cục để lưu trữ stream video và trạng thái camera
let videoStream = null;
let currentFacingMode = "environment";

const video = document.getElementById("video");
const cameraSelect = document.getElementById("cameraSelect");
const openCameraBtn = document.getElementById("openCameraBtn");
const takePhotoBtn = document.getElementById("takePhotoBtn");
const capturedPhoto = document.getElementById("capturedPhoto");
const videoContainer = document.querySelector(".video-container");
const uploadTabBtn = document.querySelector('[data-tab="uploadTab"]');
const captureTabBtn = document.querySelector('[data-tab="captureTab"]');
const uploadTab = document.getElementById("uploadTab");
const captureTab = document.getElementById("captureTab");

// Tab switching logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        if (this.dataset.tab === 'uploadTab') {
            uploadTab.classList.add('active');
            captureTab.classList.remove('active');
        } else if (this.dataset.tab === 'captureTab') {
            uploadTab.classList.remove('active');
            captureTab.classList.add('active');
            // Nếu là thiết bị di động thì tự động mở input file với capture
            if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
                const mobileCaptureInput = document.getElementById('mobileCaptureInput');
                if (mobileCaptureInput) {
                    mobileCaptureInput.value = '';
                    mobileCaptureInput.click();
                }
            }
        }
    });
});

// Mở camera
openCameraBtn.addEventListener("click", async function () {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    currentFacingMode = cameraSelect.value;

    try {
        // Thử với exact facing mode trước
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { exact: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
    } catch (e) {
        console.log("Không thể mở camera với exact mode, thử mode thường:", e);
        try {
            // Thử với facing mode thường
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
        } catch (e2) {
            console.log("Không thể mở camera, thử camera bất kỳ:", e2);
            // Thử với camera bất kỳ
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }
    }

    if (videoStream) {
        video.srcObject = videoStream;
        videoContainer.style.display = "block";
        takePhotoBtn.style.display = "inline-block";
        capturedPhoto.style.display = "none";

        // Đảm bảo video load xong
        video.onloadedmetadata = function () {
            console.log("Camera đã sẵn sàng");
        };
    } else {
        alert("Không thể mở camera. Vui lòng kiểm tra quyền truy cập camera.");
    }
});

// CHỤP ẢNH TỪ CAMERA
// Khi người dùng nhấn nút "Chụp ảnh", sẽ chụp ảnh
takePhotoBtn.addEventListener("click", function () {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const dataUrl = tempCanvas.toDataURL("image/png");
    capturedPhoto.src = dataUrl;
    capturedPhoto.style.display = "block";
    // Chuyển ảnh sang tab upload và vẽ lên canvasUpload
    uploadTabBtn.click();
    // Resize và pad ảnh như upload
    const img = new Image();
    img.onload = async function () {
        const ratio = Math.min(TARGET_SIZE / img.width, TARGET_SIZE / img.height);
        const newWidth = Math.round(img.width * ratio);
        const newHeight = Math.round(img.height * ratio);
        const canvasUpload = document.getElementById("canvasUpload");
        canvasUpload.width = TARGET_SIZE;
        canvasUpload.height = TARGET_SIZE;
        canvasUpload.style.width = "320px";
        const ctx2 = canvasUpload.getContext("2d");
        ctx2.fillStyle = "black";
        ctx2.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        const xOffset = Math.floor((TARGET_SIZE - newWidth) / 2);
        const yOffset = Math.floor((TARGET_SIZE - newHeight) / 2);
        ctx2.drawImage(img, xOffset, yOffset, newWidth, newHeight);
        showResizableBox();
        // Tạo file từ ảnh vừa chụp và gán vào fileInput
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            const res = await fetch(dataUrl);
            const blob = await res.blob();
            const file = new File([blob], 'captured_photo.png', { type: 'image/png' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        }
    };
    img.src = dataUrl;
});

// Khi người dùng chọn camera, sẽ thay đổi facing mode
captureTabBtn.addEventListener('click', function () {
    // Không tự động hiển thị video container khi chuyển tab
    // Chỉ hiển thị khi người dùng nhấn "Mở Camera"
    videoContainer.style.display = "none";
    takePhotoBtn.style.display = "none";
});

// Khi người dùng chuyển sang tab upload, sẽ ẩn video và nút chụp ảnh
uploadTabBtn.addEventListener('click', function () {
    videoContainer.style.display = "none";
    takePhotoBtn.style.display = "none";
    // Dừng camera nếu đang chạy
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
});

// Hiển thị danh sách camera
function showResultLoadingOverlay() {
    document.getElementById('resultLoadingOverlay').style.display = 'flex';
}
function hideResultLoadingOverlay() {
    document.getElementById('resultLoadingOverlay').style.display = 'none';
}
// Hook vào autoCountDiamonds và sendImageToServer
const _autoCountDiamonds = autoCountDiamonds;
autoCountDiamonds = async function (...args) {
    showResultLoadingOverlay();
    try {
        await _autoCountDiamonds.apply(this, args);
    } finally {
        hideResultLoadingOverlay();
    }
}

// Hook vào sendImageToServer để hiển thị overlay loading
// Hàm này sẽ hiển thị overlay loading khi gửi ảnh lên server
const _sendImageToServer = sendImageToServer;
sendImageToServer = async function (...args) {
    showResultLoadingOverlay();
    try {
        await _sendImageToServer.apply(this, args);
    } finally {
        hideResultLoadingOverlay();
    }
}

// Ngăn chặn sự kiện beforeunload để tránh mất dữ liệu
// Điều này sẽ hiển thị hộp thoại xác nhận khi người dùng cố gắng đóng
window.addEventListener("beforeunload", function (e) {
    e.preventDefault();
});

// ✅ Gán biến toàn cục để tránh lỗi ReferenceError
let shapeChart = null;


// Xử lý sự kiện mở modal thống kê
// Khi người dùng nhấn nút "Mở Dashboard", sẽ hiển thị modal thống
document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("statsModal");
    const closeBtn = modal.querySelector(".close");
    const openBtn = document.getElementById("openDashboardBtn");

    openBtn.addEventListener("click", () => {
        const rows = document.querySelectorAll("#resultsTableBody tr");
        let total = 0, circle = 0, triangle = 0;

        rows.forEach(row => {
            const cells = row.querySelectorAll("td");
            if (cells.length >= 4) {
                total++;
                const shape = cells[3].innerText.trim();
                if (shape === "Tròn") circle++;
                else if (shape === "Tam giác") triangle++;
            }
        });

        document.getElementById("statTotal").innerText = total;
        document.getElementById("statCircle").innerText = circle;
        document.getElementById("statTriangle").innerText = triangle;

        if (shapeChart) shapeChart.destroy();

        const ctx = document.getElementById("shapeChart").getContext("2d");
        shapeChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Hình Tròn", "Tam Giác"],
                datasets: [{
                    data: [circle, triangle],
                    backgroundColor: ["#007bff", "#dc3545"]
                }]
            }
        });

        modal.style.display = "block";
    });

    closeBtn.onclick = () => modal.style.display = "none";
    window.onclick = (e) => {
        if (e.target === modal) modal.style.display = "none";
    };
});

function updateMinSizeSliderBg() {
    const slider = document.getElementById('minSize');
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #E6266F 0%, #E6266F ${percent}%, #eee ${percent}%, #eee 100%)`;
}
document.getElementById('minSize').addEventListener('input', updateMinSizeSliderBg);
window.addEventListener('DOMContentLoaded', updateMinSizeSliderBg);

function updateSliderBg(sliderId) {
    const slider = document.getElementById(sliderId);
    if (!slider) return;
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #E6266F 0%, #E6266F ${percent}%, #eee ${percent}%, #eee 100%)`;
}

// Gọi cho cả hai slider khi load và khi thay đổi
['minSize', 'fontSize'].forEach(id => {
    const slider = document.getElementById(id);
    if (slider) {
        slider.addEventListener('input', () => updateSliderBg(id));
        // Gọi 1 lần khi load trang
        updateSliderBg(id);
    }
});

window.addEventListener('DOMContentLoaded', function () {
    var cutGroup = document.getElementById('cutOptionsGroup');
    if (cutGroup) {
        cutGroup.classList.remove('show');
        cutGroup.style.display = 'none';
    }
});

// Cập nhật giá trị hiển thị cho slider circleThreshold
const circleThresholdSlider = document.getElementById('circleThreshold');
const circleThresholdValue = document.getElementById('circleThresholdValue');
if (circleThresholdSlider && circleThresholdValue) {
    circleThresholdSlider.addEventListener('input', function () {
        circleThresholdValue.textContent = this.value;
    });
}

const circleSlider = document.getElementById('circleThreshold');
if (circleSlider) {
    updateSliderBg('circleThreshold');
    circleSlider.addEventListener('input', function () {
        updateSliderBg('circleThreshold');
    });
}

// Khi nhấn tab "Tải lên hình ảnh" trên mobile, tự động click input file
let userClickedUploadTab = false;

// Đánh dấu khi người dùng thực sự bấm vào tab upload
const uploadTabBtnEl = document.querySelector('[data-tab="uploadTab"]');
uploadTabBtnEl.addEventListener('mousedown', function () {
    userClickedUploadTab = true;
});
uploadTabBtnEl.addEventListener('touchstart', function () {
    userClickedUploadTab = true;
});

uploadTabBtnEl.addEventListener('click', function () {
    if (window.innerWidth <= 600) {
        if (userClickedUploadTab) {
            document.getElementById('fileInput').click();
        }
    }
    userClickedUploadTab = false;
});

// Hiển thị modal lưu ý khi nhấn chấm hỏi
const helpIcon = document.getElementById('helpIcon');
const helpModal = document.getElementById('helpModal');
const closeHelpModal = document.getElementById('closeHelpModal');
if (helpIcon && helpModal && closeHelpModal) {
    helpIcon.addEventListener('click', function () {
        helpModal.style.display = 'block';
    });
    closeHelpModal.addEventListener('click', function () {
        helpModal.style.display = 'none';
    });
    window.addEventListener('click', function (e) {
        if (e.target === helpModal) helpModal.style.display = 'none';
    });
}

// Xử lý khi người dùng chọn/chụp ảnh từ input file mobileCaptureInput
const mobileCaptureInput = document.getElementById('mobileCaptureInput');
if (mobileCaptureInput) {
    mobileCaptureInput.addEventListener('change', async function () {
        const file = this.files[0];
        if (file) {
            await resizeAndPadImage(file, TARGET_SIZE);
            showResizableBox();
            // Gán file vừa chụp vào fileInput để nút Tính Toán hoạt động
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                // Tạo FileList mới từ file vừa chụp
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
            }
            // Tự động chuyển về tab upload
            if (uploadTabBtn) uploadTabBtn.click();
        }
    });
}


// XỬ LÝ PHÓNG TO/THU NHỎ VÀ DI CHUYỂN ẢNH KẾT QUẢ
const zoomContainer = document.getElementById('zoomContainer');
const canvasAnnotated = document.getElementById('canvasAnnotated');
const ctxAnnotated = canvasAnnotated.getContext('2d');

let scale = 1;
let translateX = 0;
let translateY = 0;
let isPanning = false;
let startX = 0;
let startY = 0;

// Ghi đè hàm onload của ảnh để vẽ lại với các thuộc tính transform
const originalAnnotatedImageOnload = (img) => {
    canvasAnnotated.width = img.width;
    canvasAnnotated.height = img.height;
    // Đảm bảo canvas luôn vừa với khung hiển thị ban đầu
    canvasAnnotated.style.width = "100%";
    canvasAnnotated.style.height = "100%";

    // Đặt lại các giá trị zoom/pan khi ảnh mới được tải
    scale = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();

    ctxAnnotated.drawImage(img, 0, 0);
};

// Cập nhật lại logic tải ảnh trong hàm autoCountDiamonds và sendImageToServer
// để sử dụng hàm onload đã ghi đè
// Tìm và thay thế img.onload trong autoCountDiamonds
// Tìm và thay thế img.onload trong sendImageToServer

function updateTransform() {
    canvasAnnotated.style.transformOrigin = `0 0`; // Set transform origin to top-left
    canvasAnnotated.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

zoomContainer.addEventListener('wheel', (e) => {
    e.preventDefault();
    const scaleAmount = 0.1;
    const mouseX = e.clientX - zoomContainer.getBoundingClientRect().left;
    const mouseY = e.clientY - zoomContainer.getBoundingClientRect().top;

    const oldScale = scale;
    if (e.deltaY < 0) {
        scale += scaleAmount; // Zoom in
    } else {
        scale -= scaleAmount; // Zoom out
    }
    scale = Math.max(0.5, Math.min(scale, 5)); // Limit zoom level

    // Adjust translation to zoom towards the mouse pointer
    translateX = mouseX - (mouseX - translateX) * (scale / oldScale);
    translateY = mouseY - (mouseY - translateY) * (scale / oldScale);

    updateTransform();
});

zoomContainer.addEventListener('mousedown', (e) => {
    if (scale > 1) { // Only allow panning if zoomed in
        isPanning = true;
        zoomContainer.style.cursor = 'grabbing';
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
    }
});

zoomContainer.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;

    // Constrain panning to keep the image within the container
    const scaledWidth = canvasAnnotated.width * scale;
    const scaledHeight = canvasAnnotated.height * scale;
    const containerWidth = zoomContainer.clientWidth;
    const containerHeight = zoomContainer.clientHeight;

    // Prevent panning too far left
    translateX = Math.min(translateX, 0);
    // Prevent panning too far right
    translateX = Math.max(translateX, containerWidth - scaledWidth);
    // Prevent panning too far up
    translateY = Math.min(translateY, 0);
    // Prevent panning too far down
    translateY = Math.max(translateY, containerHeight - scaledHeight);

    updateTransform();
});

zoomContainer.addEventListener('mouseup', () => {
    isPanning = false;
    zoomContainer.style.cursor = 'grab';
});

zoomContainer.addEventListener('mouseleave', () => {
    isPanning = false;
    zoomContainer.style.cursor = 'grab';
});



