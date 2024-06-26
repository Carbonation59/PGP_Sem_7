#include <cstdint>

#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>


#define CSC(call) 				\
do {						\
	cudaError_t status = call;		\
	if (status != cudaSuccess) {		\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);			\
	}					\
} while(0)

typedef unsigned char uchar;
#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

// описание частицы
struct t_item {
	float x; // координата
	float y;
	float z;
	float dx; // скорость
	float dy;
	float dz;
	float q; // заряд
};

t_item* balls; // наши частицы
t_item* balls1; // для обработки картинки на гпу

t_item ball_shot;  // пуля
int n = 150;

int w = 1024;
int h = 648; // размеры экрана
float r = 0.5; // радиус шарика

float x = -1.5, y = -1.5, z = 1.0; // положение нашего игрока
float dx = 0.0, dy = 0.0, dz = 0.0;
float yaw = 0.0, pitch = 0.0; // положение нашей камеры
float dyaw = 0.0, dpitch = 0.0;
float speed = 0.15; // скорость игрока
float speed_b = 10; // скорость пули
const float a2 = 15.0; //половина стороны куба
const int np = 100; // размер текстуры пола (количество точек)
float qc = 30.0;  // заряд игрока
float qb = 50.0;   // заряд пули
bool shot = false; // маркер выстрела



GLUquadric* quadratic; // объект квадрик, орисовываем частичку в виде сферы, текстурирование сферы, с помощью объекта GLU
cudaGraphicsResource* res; // общий объект с кудой
GLuint textures[2]; // текстуры: для частицы и для пола
GLuint vbo; // номер под опенжлевский буфер

void display() { // функция перересовки графики
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // очищаем буффер вывода

	glMatrixMode(GL_PROJECTION); // сбрасываем матрицу проекций
	glLoadIdentity();
	gluPerspective(90.0f, (GLfloat)w / (GLfloat)h, 0.1f, 100.0f); // задаём матрицу перспективного преобразования аргументы:(угол обзора, соотношение сторон, задаются параметры усеченной пирамиды обзора)

	// начинаем работу с матрицей моделей вида, сначала её сбрасываем
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// задаём эту матрицу (как работает наша камера) аргументы: (положение камеры, куда смотрим камерв, направление наверх (где вверх камеры))
	gluLookAt(x, y, z,
		x + cos(yaw) * cos(pitch),
		y + sin(yaw) * cos(pitch),
		z + sin(pitch),
		0.0f, 0.0f, 1.0f);

	// делаем активным нулевую текстуру (для сферы)
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	// реализуем вращение
	static float angle = 0.0;
	for (int i = 0;i < n;i++) {
		glPushMatrix(); // создаём новую матрицу
		glTranslatef(balls[i].x, balls[i].y, balls[i].z); // перемещаем объект в те координаты, в которых он находится
		glRotatef(angle, 0.0, 0.0, 1.0); // вращаем
		gluSphere(quadratic, r, 32, 32); // отрисовываем сферу, задаём радиус сферы и количество разбиений по широте и долготе (сколько полигонов)
		// возвращаемся к предыдущей матрице
		glPopMatrix();
	}
	if (shot) {
		glPushMatrix();
		glTranslatef(ball_shot.x, ball_shot.y, ball_shot.z);
		glRotatef(angle, 0.0, 0.0, 1.0);
		gluSphere(quadratic, r, 32, 32);
		glPopMatrix();
	}
	angle += 0.15; // изменяем угол поворота

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo); // делаем активным буффер для пола
	glBindTexture(GL_TEXTURE_2D, textures[1]); // делаем активным соответствующую текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); // берем данные из активного буффера для цветов текстуры
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // деактивируем буффер
	glBegin(GL_QUADS); // рисуем пол (рисуем квадрат)
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-a2, -a2, 0.0);

	glTexCoord2f(1.0, 0.0);
	glVertex3f(a2, -a2, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(a2, a2, 0.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(-a2, a2, 0.0);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0); // делаем текстуру не активной

	glLineWidth(2); // толщина линий
	glColor3f(0.5f, 0.5f, 0.5f); // цвет линий
	glBegin(GL_LINES); // отрисовываем скелет куба
	glVertex3f(-a2, -a2, 0.0);
	glVertex3f(-a2, -a2, 2.0 * a2);

	glVertex3f(a2, -a2, 0.0);
	glVertex3f(a2, -a2, 2.0 * a2);

	glVertex3f(a2, a2, 0.0);
	glVertex3f(a2, a2, 2.0 * a2);

	glVertex3f(-a2, a2, 0.0);
	glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glBegin(GL_LINE_LOOP); // рисуем пол  (соединяем точки замкнутой линией)
	glVertex3f(-a2, -a2, 0.0);
	glVertex3f(a2, -a2, 0.0);
	glVertex3f(a2, a2, 0.0);
	glVertex3f(-a2, a2, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP); // и потолок
	glVertex3f(-a2, -a2, 2.0 * a2);
	glVertex3f(a2, -a2, 2.0 * a2);
	glVertex3f(a2, a2, 2.0 * a2);
	glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();

}

__global__ void kernel(uchar4* data, t_item* balls, t_item ball_shot, bool shot, float t, int n) { //генерация текстуры пола на GPU
	float a2 = 15.0;
	float K = 50.0;
	float shift = 0.75; // сдвиг для карты напряжённости
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	float x, y, fg;
	fg = 0;
	for (i = idx; i < np; i += offsetx)
		for (j = idy; j < np; j += offsety) {
			x = (2.0 * i / (np - 1.0) - 1.0) * a2; // переход из координат пикселей в координаты физические
			y = (2.0 * j / (np - 1.0) - 1.0) * a2;
			for (int l = 0;l < n;l++) {
				fg = fg + balls[l].q / (sqr(x - balls[l].x) + sqr(y - balls[l].y) + sqr(balls[l].z - shift) + 0.001); // учитываем напряженность
			}
			if (shot) {
				fg = fg + ball_shot.q / (sqr(x - ball_shot.x) + sqr(y - ball_shot.y) + sqr(ball_shot.z - shift) + 0.001);
			}
			fg = fg * K;
			data[j * np + i] = make_uchar4(0, 0, min(int(fg), 255), 255);
		}
}

// реализация физики изменения координат и положения частичек
void update() {
	//dz = dz - 0.0001;   // реализация гравитации

	float v = sqrt(dx * dx + dy * dy + dz * dz);  //абсолютная скорость движения игрока
	if (v > speed) {					//  Ограничение максимальной скорости
		dx = dx * speed / v;
		dy = dy * speed / v;
		dz = dz * speed / v;
	}

	// перемещение
	x = x + dx;
	y = y + dy;
	z = z + dz;

	// инерция
	dx = dx * 0.99;
	dy = dy * 0.99;
	dz = dz * 0.99;

	// условие, чтобы не проваливаться сквозь пол
	if (z < 1.0) {
		z = 1.0;
		dz = 0.0;
	}

	// убрать медленные вращения
	if (fabs(dpitch) + fabs(dyaw) > 0.0001) {
		yaw = yaw + dyaw;
		pitch = pitch + dpitch;
		pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, pitch)); // ограничиваем угол поворота
		dyaw = 0.0;	// отключаем инерцию мышки
		dpitch = 0.0;
	}

	float w = 0.99;   // коэф. замедления
	float e0 = 1e-3;
	float dt = 0.01;   // шаг по времени
	float K = 50.0;
	float g = 20.0;   // коэф. гравитации

	for (int i = 0;i < n;i++) {
		// замедление частицы
		balls[i].dx = balls[i].dx * w;
		balls[i].dy = balls[i].dy * w;
		balls[i].dz = balls[i].dz * w;

		// отталкивание от стен
		balls[i].dx = balls[i].dx + balls[i].q * balls[i].q * K * (balls[i].x - a2) / (sqr3(fabs(balls[i].x - a2)) + e0) * dt;
		balls[i].dx = balls[i].dx + balls[i].q * balls[i].q * K * (balls[i].x + a2) / (sqr3(fabs(balls[i].x + a2)) + e0) * dt;

		balls[i].dy = balls[i].dy + balls[i].q * balls[i].q * K * (balls[i].y - a2) / (sqr3(fabs(balls[i].y - a2)) + e0) * dt;
		balls[i].dy = balls[i].dy + balls[i].q * balls[i].q * K * (balls[i].y + a2) / (sqr3(fabs(balls[i].y + a2)) + e0) * dt;

		balls[i].dz = balls[i].dz + balls[i].q * balls[i].q * K * (balls[i].z - 2.0 * a2) / (sqr3(fabs(balls[i].z - 2.0 * a2)) + e0) * dt;
		balls[i].dz = balls[i].dz + balls[i].q * balls[i].q * K * (balls[i].z + 0.0) / (sqr3(fabs(balls[i].z + 0.0)) + e0) * dt;

		balls[i].dz = balls[i].dz - g * dt;

		float l = sqrt(sqr(balls[i].x - x) + sqr(balls[i].y - y) + sqr(balls[i].z - z));  // расстояние от частички до камеры

		// отталкивание от камеры
		balls[i].dx = balls[i].dx + qc * balls[i].q * K * (balls[i].x - x) / (l * l * l + e0) * dt;
		balls[i].dy = balls[i].dy + qc * balls[i].q * K * (balls[i].y - y) / (l * l * l + e0) * dt;
		balls[i].dz = balls[i].dz + qc * balls[i].q * K * (balls[i].z - z) / (l * l * l + e0) * dt;

		if (shot) {
			// расстояние от пули до камеры

			l = sqrt(sqr(balls[i].x - ball_shot.x) + sqr(balls[i].y - ball_shot.y) + sqr(balls[i].z - ball_shot.z));    

			// отталкивание от пули

			balls[i].dx = balls[i].dx + qb * balls[i].q * K * (balls[i].x - ball_shot.x) / (l * l * l + e0) * dt;
			balls[i].dy = balls[i].dy + qb * balls[i].q * K * (balls[i].y - ball_shot.y) / (l * l * l + e0) * dt;
			balls[i].dz = balls[i].dz + qb * balls[i].q * K * (balls[i].z - ball_shot.z) / (l * l * l + e0) * dt;
		}

		//отталкивание между шарами

		for (int j = 0;j < n;j++) {
			l = sqrt(sqr(balls[i].x - balls[j].x) + sqr(balls[i].y - balls[j].y) + sqr(balls[i].z - balls[j].z));
			balls[i].dx = balls[i].dx + balls[j].q * balls[i].q * K * (balls[i].x - balls[j].x) / (l * l * l + e0) * dt;
			balls[i].dy = balls[i].dy + balls[j].q * balls[i].q * K * (balls[i].y - balls[j].y) / (l * l * l + e0) * dt;
			balls[i].dz = balls[i].dz + balls[j].q * balls[i].q * K * (balls[i].z - balls[j].z) / (l * l * l + e0) * dt;
		}

		// изменяем положение частички
		balls[i].x = balls[i].x + balls[i].dx * dt;
		balls[i].y = balls[i].y + balls[i].dy * dt;
		balls[i].z = balls[i].z + balls[i].dz * dt;

		// иногда (довольно редко) шары вылетают за пределы куба, поэтому поставим такую проверку
		balls[i].x = fmin(fmax(balls[i].x, -a2 + r + e0), a2 - r - e0);
		balls[i].y = fmin(fmax(balls[i].y, -a2 + r + e0), a2 - r - e0);
		balls[i].z = fmin(fmax(balls[i].z, r + e0), 2 * a2 - r - e0);

	}
	if (shot) {
		ball_shot.x = ball_shot.x + ball_shot.dx * dt;
		ball_shot.y = ball_shot.y + ball_shot.dy * dt;
		ball_shot.z = ball_shot.z + ball_shot.dz * dt;
	}
	static float t = 0.0;		// шаг по времени
	uchar4* dev_data;
	size_t size;
	cudaGraphicsMapResources(1, &res, 0);		// делаем буфер доступным для CUDA
	cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res);	// Получаем указатель на память
	cudaMemcpy(balls1, balls, sizeof(t_item) * n, cudaMemcpyHostToDevice);
	kernel << <dim3(32, 32), dim3(32, 8) >> > (dev_data, balls1, ball_shot, shot, t, n);		// 
	cudaGraphicsUnmapResources(1, &res, 0);			// Возвращаем указатель OpenGL'ю чтобы он мог его использовать
	t += 0.01;

	glutPostRedisplay();
}

void keys(unsigned char key, int x, int y) {// обрабатываем нажатия на кнопки
	switch (key) {
		// изменяем направление движения (вектор скорости), относительно направления нашей камеры
	case 'w': // движение вперед
		dx += cos(yaw) * cos(pitch) * speed;
		dy += sin(yaw) * cos(pitch) * speed;
		dz += sin(pitch) * speed;
		break;
	case 's': // назад
		dx += -cos(yaw) * cos(pitch) * speed;
		dy += -sin(yaw) * cos(pitch) * speed;
		dz += -sin(pitch) * speed;
		break;
	case 'a': // влево
		dx += -sin(yaw) * speed;
		dy += cos(yaw) * speed;
		break;
	case 'd': // вправо
		dx += sin(yaw) * speed;
		dy += -cos(yaw) * speed;
		break;
	case 27: // завершаем приложение
		cudaGraphicsUnregisterResource(res);
		glDeleteTextures(2, textures);
		glDeleteBuffers(1, &vbo);
		gluDeleteQuadric(quadratic);
		free(balls);
		cudaFree(balls1);
		exit(0);
		break;
	}
}

void mouse(int x, int y) { // что происходит, при движении мыши?
	static int x_prev = w / 2, y_prev = h / 2; // предыдущие координаты нашей позиции
	float dx1 = 0.005 * (x - x_prev); // определяем поворот камеры
	float dy1 = 0.005 * (y - y_prev);
	dyaw -= dx1; // учитываем это, изменяя скорости поворотов наших углов камеры
	dpitch -= dy1;
	x_prev = x; // запоминаем предыдущую позицию
	y_prev = y;
	if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) { // если указатель мыши слишком близко к границам (если слишком резко дернем за мышкку, выйдем за пределы окна)
		glutWarpPointer(w / 2, h / 2); // перемещаем указатель в центр нашего экрана
		x_prev = w / 2; // запоминаем предыдущие значения нашего экрана
		y_prev = h / 2;
	}
}

void mouse_shot(int button, int state, int x1, int y1) { // что происходит, при движении мыши?
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			shot = true;
			// вектор скорости пули 
			ball_shot.dx = speed_b * cos(yaw) * cos(pitch);
			ball_shot.dy = speed_b * sin(yaw) * cos(pitch);
			ball_shot.dz = speed_b * sin(pitch);
			// начальные координаты пули
			ball_shot.x = x + ball_shot.dx;
			ball_shot.y = y + ball_shot.dy;
			ball_shot.z = z + ball_shot.dz;
			//заряд пули
			ball_shot.q = qb;
		}
	}

}

void reshape(int w_new, int h_new) { // что происходит, когда меняем размер экрана?
	w = w_new; // запоминаем новые значения размеров экрана
	h = h_new;
	glViewport(0, 0, w, h); // сброс текущей области вывода
	glMatrixMode(GL_PROJECTION); // выбираем матрицу проекций
	glLoadIdentity(); // также делаем сброс этой матрицы
}

int main(int argc, char** argv) {
	balls = (t_item*)malloc(sizeof(t_item) * n);
	cudaMalloc(&balls1, sizeof(t_item) * n);
	// инициализоация глута
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);   // обновление положения игрока, частичек
	glutDisplayFunc(display);  // функция для отрисовки
	glutKeyboardFunc(keys);    // функция для обработки нажатия на кнопки
	glutPassiveMotionFunc(mouse);  // функция для обработки движения мышки
	glutMouseFunc(mouse_shot);       // функция для обработки щелчка мышкой
	glutReshapeFunc(reshape);	// функция для изменения размеров экрана

	glutSetCursor(GLUT_CURSOR_NONE); // скрыть курсор

	// загрузка текстуры Земли
	int wt, ht;
	FILE* in = fopen("in.data", "rb");
	fread(&wt, sizeof(int), 1, in);
	fread(&ht, sizeof(int), 1, in);
	uchar* data = (uchar*)malloc(sizeof(uchar) * wt * ht * 4);
	fread(data, sizeof(uchar), 4 * wt * ht, in);
	fclose(in);

	// загружаем данные в текстурную память
	glGenTextures(2, textures);  // в массиве textures генерируем текстуры
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data); // задаются параметры текстуры
	free(data);

	// задаём политику обработки (задаём, чтобы интерполяция не применялась)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//создаём объект сферы
	quadratic = gluNewQuadric();
	gluQuadricTexture(quadratic, GL_TRUE); // она у нас будет текстурироваться

	glBindTexture(GL_TEXTURE_2D, textures[1]); // задаём параметры пола
	// дальше идёт работа со второй текстурой, т.к. OpenGL - это конечный автомат
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // задаём интерполяцию (сглаживание) для пола 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glEnable(GL_TEXTURE_2D);	// Разрешить наложение текстуры
	glShadeModel(GL_SMOOTH);	// Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);	// Чёрный фон
	glClearDepth(1.0f);			// Установка буфера глубины
	glDepthFunc(GL_LEQUAL);			// Тип теста глубины.
	glEnable(GL_DEPTH_TEST);		// Включаем тест глубины
	glEnable(GL_CULL_FACE);			// Режим, при котором текстуры накладываются только с одной стороны

	// инициализация буфера
	glewInit();
	glGenBuffers(1, &vbo);		// генерируем буфер
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);	// делаем буфер активным
	glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// задаём размер буфера
	cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);		// регистрируем буфер в куде
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);		// деактивируем данный буфер

	for (int i = 0;i < n;i++) {
		//исходное положение
		balls[i].x = (rand() % int(a2 * 2 - 2 * r)) - (a2 - 2 * r);
		balls[i].y = (rand() % int(a2 * 2 - 2 * r)) - (a2 - 2 * r);
		balls[i].z = (rand() % int(2 * a2 - 4 * r)) + 2 * r;
		//параметры скорости
		balls[i].dx = 0.0;
		balls[i].dy = 0.0;
		balls[i].dz = 0.0;
		//заряд
		balls[i].q = 1.0;
	}

	glutMainLoop();
}
