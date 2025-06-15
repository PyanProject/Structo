// Расширенный загрузчик OBJ для поддержки вершинных цветов
class CustomOBJLoader extends THREE.OBJLoader {
    parse(text) {
        // Вызываем оригинальный парсер
        const object = super.parse(text);
        
        // Проверяем, содержит ли OBJ файл вершинные цвета
        const lines = text.split('\n');
        const hasVertexColors = lines.some(line => {
            const parts = line.trim().split(/\s+/);
            return parts[0] === 'v' && parts.length >= 7; // v x y z r g b
        });
        
        if (hasVertexColors) {
            console.log('Обнаружены вершинные цвета в OBJ файле, применяем их');
            
            // Обрабатываем все меши в объекте
            object.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    // Создаем буфер для цветов
                    const geometry = child.geometry;
                    const positionAttribute = geometry.getAttribute('position');
                    const count = positionAttribute.count;
                    
                    // Создаем атрибут цвета, если его еще нет
                    if (!geometry.hasAttribute('color')) {
                        const colors = [];
                        const vertices = [];
                        
                        // Собираем все вершины с их цветами из OBJ файла
                        lines.forEach(line => {
                            const parts = line.trim().split(/\s+/);
                            if (parts[0] === 'v' && parts.length >= 7) {
                                // v x y z r g b
                                vertices.push({
                                    position: new THREE.Vector3(
                                        parseFloat(parts[1]),
                                        parseFloat(parts[2]),
                                        parseFloat(parts[3])
                                    ),
                                    color: new THREE.Color(
                                        parseFloat(parts[4]),
                                        parseFloat(parts[5]),
                                        parseFloat(parts[6])
                                    )
                                });
                            }
                        });
                        
                        // Для каждой вершины в геометрии находим соответствующий цвет
                        for (let i = 0; i < count; i++) {
                            const x = positionAttribute.getX(i);
                            const y = positionAttribute.getY(i);
                            const z = positionAttribute.getZ(i);
                            
                            // Ищем ближайшую вершину в исходном файле
                            let minDist = Infinity;
                            let closestColor = new THREE.Color(1, 1, 1);
                            
                            vertices.forEach(vertex => {
                                const dist = Math.sqrt(
                                    Math.pow(vertex.position.x - x, 2) +
                                    Math.pow(vertex.position.y - y, 2) +
                                    Math.pow(vertex.position.z - z, 2)
                                );
                                
                                if (dist < minDist) {
                                    minDist = dist;
                                    closestColor = vertex.color;
                                }
                            });
                            
                            colors.push(closestColor.r, closestColor.g, closestColor.b);
                        }
                        
                        // Добавляем атрибут цвета в геометрию
                        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                    }
                    
                    // Применяем материал с вершинными цветами
                    child.material = new THREE.MeshPhongMaterial({
                        vertexColors: true,
                        specular: 0x111111,
                        shininess: 25
                    });
                }
            });
        }
        
        return object;
    }
} 