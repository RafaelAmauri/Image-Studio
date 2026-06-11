all:
	docker compose up --build
	docker compose exec image-studio bash

clean:
	docker compose down
