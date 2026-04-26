typedef void (*cheesefunc)(char *name, void *user_data);
void find_cheeses(cheesefunc user_func, void *user_data);

typedef int (*cheese_progress_callback)(float progress, void * user_data);

typedef struct _cheese_params {
	int age;
	cheese_progress_callback progress_callback;
	void * user_data;
} cheese_params;
