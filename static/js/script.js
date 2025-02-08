$(document).ready(function() {
    // 获取组件列表
    $.get('/get_components', function(data) {
        // 加载数据集
        data.datasets.forEach(function(dataset) {
            // 确保使用正确的数据集名称格式
            let datasetName = dataset;
            if (dataset === 'FashionMNIST') {
                datasetName = 'FashionMNIST';  // 保持原始格式
            }
            $('#datasets .draggable-section').append(`
                <div class="draggable" data-type="dataset">
                    <span class="component-name">${datasetName}</span>
                </div>
            `);
        });

        // 加载模型层
        data.modules.forEach(function(module) {
            $('#modules .draggable-section').append(`
                <div class="draggable config-component" data-type="module">
                    <span class="component-name">${module}</span>
                    ${module === 'Conv2D' ? `
                        <div class="config-options" style="display: none;">
                            <div class="form-group">
                                <label>输出通道数</label>
                                <input type="number" class="form-control" name="out_channels" value="32">
                            </div>
                            <div class="form-group">
                                <label>卷积核大小</label>
                                <input type="number" class="form-control" name="kernel_size" value="3">
                            </div>
                        </div>
                    ` : ''}
                    ${module === 'Linear' ? `
                        <div class="config-options" style="display: none;">
                            <div class="form-group">
                                <label>输出特征数</label>
                                <input type="number" class="form-control" name="out_features" value="10">
                            </div>
                        </div>
                    ` : ''}
                    ${module === 'Dropout' ? `
                        <div class="config-options" style="display: none;">
                            <div class="form-group">
                                <label>丢弃率</label>
                                <input type="number" class="form-control" name="p" value="0.5" step="0.1" min="0" max="1">
                            </div>
                        </div>
                    ` : ''}
                </div>
            `);
        });

        // 加载已保存的模型
        data.saved_models.forEach(function(model) {
            $('#saved-models .draggable-section').append(`
                <div class="saved-model">${model}</div>
            `);
        });

        // 初始化拖拽
        $('.draggable').draggable({
            helper: 'clone'
        });

        // 放置区域
        $('#drop-area').droppable({
            accept: '.draggable',
            drop: function(event, ui) {
                var type = ui.draggable.data('type');
                var name = ui.draggable.find('.component-name').text();
                var options = ui.draggable.find('.config-options').clone(true);
                
                // 如果是数据集类型，先移除已有的数据集组件
                if (type === 'dataset') {
                    $('#drop-area .dropped-component[data-type="dataset"]').remove();
                }
                
                // 如果是预处理组件，确保只有一个
                if (type === 'augmentation' || type === 'normalization') {
                    $(`#drop-area .dropped-component[data-type="${type}"]`).remove();
                }
                
                var component = $(`
                    <div class="dropped-component" data-type="${type}">
                        <span class="drag-handle">⋮</span>
                        <span class="component-name">${name}</span>
                        <button class="remove-btn" title="删除">×</button>
                    </div>
                `);
                
                if (options.length) {
                    component.append(options);
                }
                
                $(this).append(component);
                setTimeout(visualizeModel, 100);
            }
        });

        // 使放置的组件可排序
        $('#drop-area').sortable({
            handle: '.drag-handle',
            update: function(event, ui) {
                visualizeModel();
            }
        });

        // 删除组件
        $(document).on('click', '.remove-btn', function() {
            $(this).parent().remove();
            visualizeModel();
        });
    });

    // 添加模型结构可视化
    function visualizeModel() {
        var layers = [];
        $('#drop-area .dropped-component').each(function() {
            layers.push($(this).find('.component-name').text());
        });
        
        if (layers.length > 0) {
            var svg = d3.select("#model-visualization")
                .html("") // 清空现有内容
                .append("svg")
                .attr("width", 600)
                .attr("height", layers.length * 60);

            // 绘制每一层
            layers.forEach((layer, i) => {
                var g = svg.append("g")
                    .attr("transform", `translate(50, ${i * 60 + 30})`);
                
                // 绘制节点
                g.append("rect")
                    .attr("width", 500)
                    .attr("height", 40)
                    .attr("rx", 5)
                    .attr("ry", 5)
                    .attr("class", "layer-node");
                
                // 添加文字
                g.append("text")
                    .attr("x", 250)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .text(layer);
                
                // 如果不是最后一层，添加连接线
                if (i < layers.length - 1) {
                    svg.append("path")
                        .attr("d", `M300,${i * 60 + 70} L300,${i * 60 + 90}`)
                        .attr("stroke", "#999")
                        .attr("stroke-width", 2)
                        .attr("marker-end", "url(#arrow)");
                }
            });
        }
    }

    // 监听组件拖放，实时更新可视化
    $('#drop-area').on('drop', function() {
        setTimeout(visualizeModel, 100); // 延迟一下等待DOM更新
    });

    // 添加配置组件的点击展开/收起
    $(document).on('click', '.config-component .component-name', function(e) {
        e.stopPropagation();
        $(this).siblings('.config-options').slideToggle();
    });

    // 点击其他地方时收起配置选项
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.config-component').length) {
            $('.config-options').slideUp();
        }
    });

    // 帮助按钮点击事件
    $('#help-button').click(function() {
        $('#help-panel').toggleClass('show');
    });

    // 关闭按钮点击事件
    $('.close-help-button').click(function() {
        $('#help-panel').removeClass('show');
    });
});
